#!/usr/bin/env python
"""Build an outcome-blind cross-cohort feature lock for therapy prioritization."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXPRESSION = ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival"
LANDSCAPE = ROOT / "04_analysis/08_biological_landscape"
GMT = ROOT / "02_data/01_raw/reference/msigdb/h.all.v2026.1.Hs.symbols.gmt"
PATHWAYS = ROOT / "04_analysis/11_drug_repositioning/06_collapsed_state_pathway_map/state_pathway_representatives_annotated.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/20_therapy_prioritization_feature_lock"

CONFIG = {
    "TCGA_BRCA": {
        "expression": EXPRESSION / "TCGA_BRCA_expression.h5",
        "axes": LANDSCAPE / "tcga_emt_immune/tcga_emt_immune_axes_and_states.tsv.gz",
    },
    "METABRIC": {
        "expression": EXPRESSION / "METABRIC_expression.h5",
        "axes": LANDSCAPE / "bulk_replication/METABRIC/emt_immune_axes_and_states.tsv.gz",
    },
    "SCANB_GSE96058": {
        "expression": EXPRESSION / "SCANB_GSE96058_expression.h5",
        "axes": LANDSCAPE / "bulk_replication/SCANB_GSE96058/emt_immune_axes_and_states.tsv.gz",
    },
}

HALLMARKS = {
    "pi3k_akt_mtor": "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "mtorc1": "HALLMARK_MTORC1_SIGNALING",
    "angiogenesis": "HALLMARK_ANGIOGENESIS",
    "hypoxia": "HALLMARK_HYPOXIA",
    "tgfb": "HALLMARK_TGF_BETA_SIGNALING",
    "emt_hallmark": "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "il6_jak_stat3": "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "inflammatory_response": "HALLMARK_INFLAMMATORY_RESPONSE",
    "ifng_response": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
}

AXES = [
    "epithelial", "mesenchymal_core", "emt_position", "em_coactivation",
    "antigen_presentation", "priming_recruitment", "infiltration", "execution",
    "suppressive_recruitment", "overall_activation", "immune_exclusion_pressure",
    "stromal_z", "immune_state", "pam50",
]


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def load_gmt(path: Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sets[fields[0]] = sorted({gene.upper() for gene in fields[2:] if gene})
    return sets


def erbb_module() -> list[str]:
    frame = pd.read_csv(PATHWAYS, sep="\t", low_memory=False)
    hit = frame[
        frame["representative"].astype(str).str.casefold().eq("true")
        & frame["bulk_state_class"].eq("hybrid-peak")
        & frame["pathway"].eq("ERBB2 Activates PTK6 Signaling")
    ]
    if len(hit) != 1:
        raise ValueError(f"Expected one frozen ERBB2-PTK6 representative, found {len(hit)}")
    return sorted({x.upper() for x in str(hit.iloc[0]["detected_gene_symbols"]).split(";") if x})


def load_expression(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        x = handle["X"][...].astype(np.float32)
        samples = decode(handle["sample_id"][...])
        genes = [x.upper() for x in decode(handle["gene_symbol"][...])]
    if len(set(genes)) != len(genes):
        raise ValueError(f"Duplicate gene symbols in {path}")
    return x, samples, genes


def score_modules(
    x: np.ndarray,
    samples: list[str],
    genes: list[str],
    modules: dict[str, list[str]],
    cohort: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    means = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    variable = np.isfinite(sd) & (sd > 0)
    x[:, variable] = (x[:, variable] - means[variable]) / sd[variable]
    x[:, ~variable] = 0
    lookup = {gene: index for index, gene in enumerate(genes)}
    result = {"sample_id": samples}
    manifest: list[dict[str, object]] = []
    for module, source_genes in modules.items():
        measured = [gene for gene in source_genes if gene in lookup]
        coverage = len(measured) / len(source_genes)
        minimum = 8 if module == "erbb2_ptk6" else 10
        passed = len(measured) >= minimum and coverage >= 0.70
        manifest.append({
            "cohort": cohort,
            "feature": module,
            "source_genes": len(source_genes),
            "measured_genes": len(measured),
            "coverage": coverage,
            "minimum_genes": minimum,
            "coverage_pass": passed,
            "measured_gene_symbols": ";".join(measured),
        })
        if not passed:
            raise ValueError(f"Insufficient {module} coverage in {cohort}: {len(measured)}/{len(source_genes)}")
        result[module] = np.mean(x[:, [lookup[gene] for gene in measured]], axis=1)
    return pd.DataFrame(result), manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gmt = load_gmt(GMT)
    modules = {name: gmt[key] for name, key in HALLMARKS.items()}
    modules["erbb2_ptk6"] = erbb_module()
    missing = [key for key in HALLMARKS.values() if key not in gmt]
    if missing:
        raise ValueError(f"Missing locked Hallmark modules: {missing}")

    module_rows = []
    for module, genes in modules.items():
        for gene in genes:
            module_rows.append({
                "feature": module,
                "gene": gene,
                "source": "study_hybrid_representative" if module == "erbb2_ptk6" else "MSigDB_hallmark_v2026.1",
            })
    pd.DataFrame(module_rows).to_csv(OUT / "frozen_therapy_feature_gene_sets.tsv", sep="\t", index=False)

    manifests: list[dict[str, object]] = []
    correlations = []
    cohort_summary = {}
    for cohort, config in CONFIG.items():
        x, samples, genes = load_expression(config["expression"])
        scores, coverage = score_modules(x, samples, genes, modules, cohort)
        manifests.extend(coverage)
        axes = pd.read_csv(config["axes"], sep="\t", low_memory=False)
        required = ["sample_id", *AXES]
        missing_axes = sorted(set(required) - set(axes.columns))
        if missing_axes:
            raise ValueError(f"Missing axes in {cohort}: {missing_axes}")
        if axes["sample_id"].duplicated().any() or scores["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {cohort}")
        merged = axes[required].merge(scores, on="sample_id", how="inner", validate="one_to_one")
        if len(merged) != len(axes) or len(merged) != len(scores):
            raise ValueError(f"Sample linkage failure in {cohort}: axes={len(axes)}, scores={len(scores)}, merged={len(merged)}")
        merged.insert(1, "cohort", cohort)
        merged.to_csv(OUT / f"{cohort}_therapy_prioritization_features.tsv.gz", sep="\t", index=False, compression="gzip")

        numeric = [x for x in [*modules.keys(), *AXES] if x in merged and pd.api.types.is_numeric_dtype(merged[x])]
        corr = merged[numeric].corr(method="spearman")
        for left in modules:
            for right in numeric:
                if left != right:
                    correlations.append({
                        "cohort": cohort,
                        "feature_1": left,
                        "feature_2": right,
                        "spearman_rho": float(corr.loc[left, right]),
                    })
        cohort_summary[cohort] = {
            "samples": int(len(merged)),
            "features_locked": int(len(modules)),
            "immune_states": {str(k): int(v) for k, v in merged["immune_state"].value_counts().items()},
        }

    coverage_frame = pd.DataFrame(manifests)
    coverage_frame.to_csv(OUT / "therapy_feature_gene_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "therapy_feature_spearman_correlations.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind cross-cohort therapy-prioritization feature lock",
        "scoring": "within-cohort gene z scores followed by unweighted gene-set means",
        "cohorts": cohort_summary,
        "locked_features": sorted(modules),
        "all_coverage_gates_passed": bool(coverage_frame["coverage_pass"].all()),
        "outcomes_used": False,
        "therapy_weights_fitted": False,
        "next_gate": "freeze interpretable therapy-hypothesis formulas and quantify cross-cohort stability under weight sensitivity analysis",
    }
    (OUT / "therapy_prioritization_feature_lock_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
