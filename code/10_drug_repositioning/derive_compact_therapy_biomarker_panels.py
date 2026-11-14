#!/usr/bin/env python
"""Derive in TCGA and externally validate compact, outcome-blind therapy panels."""
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
from scipy.stats import rankdata, spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXPR = ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival"
FEATURES = ROOT / "04_analysis/11_drug_repositioning/20_therapy_prioritization_feature_lock"
SCORES = ROOT / "04_analysis/11_drug_repositioning/21_state_specific_therapy_scores"
OUT = ROOT / "04_analysis/11_drug_repositioning/22_compact_therapy_biomarker_panels"
COHORTS = ["TCGA_BRCA", "METABRIC", "SCANB_GSE96058"]
DISCOVERY = "TCGA_BRCA"
EPS = 1e-6

E_CORE = ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "OCLN"]
M_CORE = ["VIM", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "FN1", "ITGA5"]
IMMUNE_VISIBILITY = [
    "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "NLRC5", "PSMB8", "PSMB9",
    "STAT1", "IRF1", "GBP1", "CXCL9", "CXCL10", "CXCL11", "IDO1", "IFIT1", "IFIT3", "ISG15",
    "CCL5", "CCL4", "XCL1", "XCL2",
]
INCOMPLETE = [
    "CD274", "PDCD1LG2", "LGALS9", "PVR", "NECTIN2", "CD80", "CD86",
    "CCL2", "CSF1", "CSF2", "CXCL1", "CXCL2", "CXCL3", "CXCL5", "CXCL8", "CCL20",
    "TGFB1", "TGFB2", "TGFBI", "SERPINE1", "CTGF", "FN1", "COL1A1", "COL1A2", "ITGA5", "LOX",
]


def decode(values):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def load_expression(cohort: str) -> tuple[pd.DataFrame, list[str]]:
    with h5py.File(EXPR / f"{cohort}_expression.h5", "r") as handle:
        x = handle["X"][...].astype(np.float32)
        samples = decode(handle["sample_id"][...])
        genes = [g.upper() for g in decode(handle["gene_symbol"][...])]
    return pd.DataFrame(x, index=samples, columns=genes), genes


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    sd = frame.std(axis=0, ddof=0).replace(0, np.nan)
    return ((frame - mean) / sd).fillna(0.0)


def pct(x: pd.Series) -> pd.Series:
    return x.rank(method="average", pct=True).clip(EPS, 1.0)


def geomean(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.exp(np.log(frame.clip(EPS, 1.0)).mean(axis=1)), index=frame.index)


def load_gene_sets() -> dict[str, list[str]]:
    manifest = pd.read_csv(FEATURES / "frozen_therapy_feature_gene_sets.tsv", sep="\t")
    return {name: group["gene"].astype(str).str.upper().tolist() for name, group in manifest.groupby("feature")}


def target_domains(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rank = lambda c: pct(feature_frame[c])
    return pd.DataFrame({
        "hybrid_state": rank("em_coactivation"),
        "erbb_dependency": rank("erbb2_ptk6"),
        "akt_pathway": rank("pi3k_akt_mtor"),
        "mtor_pathway": rank("mtorc1"),
        "mesenchymal_state": pd.concat([rank("mesenchymal_core"), rank("emt_hallmark")], axis=1).mean(axis=1),
        "vascular_stromal_barrier": pd.concat([rank("angiogenesis"), rank("hypoxia"), rank("tgfb"), rank("stromal_z")], axis=1).mean(axis=1),
        "immune_visibility": pd.concat([rank("antigen_presentation"), rank("priming_recruitment"), rank("infiltration"), rank("ifng_response")], axis=1).mean(axis=1),
        "incomplete_completion": pd.concat([rank("immune_exclusion_pressure"), rank("suppressive_recruitment")], axis=1).mean(axis=1),
    }, index=feature_frame.index)


def select_panel(z: pd.DataFrame, target: pd.Series, candidates: list[str], k: int) -> list[tuple[str, float]]:
    rows = []
    y = rankdata(target.to_numpy(float))
    for gene in sorted(set(candidates) & set(z.columns)):
        rho = float(np.corrcoef(rankdata(z[gene].to_numpy(float)), y)[0, 1])
        if np.isfinite(rho):
            rows.append((gene, rho))
    positive = sorted([x for x in rows if x[1] > 0], key=lambda x: (-x[1], x[0]))
    if len(positive) < k:
        raise ValueError(f"Only {len(positive)} positive candidates for panel of {k}")
    return positive[:k]


def panel_domain(z: pd.DataFrame, genes: list[str]) -> pd.Series:
    missing = sorted(set(genes) - set(z.columns))
    if missing:
        raise ValueError(f"External cohort missing locked panel genes: {missing}")
    return pct(z[genes].mean(axis=1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = load_gene_sets()
    candidates = {
        "hybrid_state": E_CORE + M_CORE,
        "erbb_dependency": sets["erbb2_ptk6"],
        "akt_pathway": sets["pi3k_akt_mtor"],
        "mtor_pathway": sets["mtorc1"],
        "mesenchymal_state": sorted(set(M_CORE + sets["emt_hallmark"])),
        "vascular_stromal_barrier": sorted(set(sets["angiogenesis"] + sets["hypoxia"] + sets["tgfb"])),
        "immune_visibility": IMMUNE_VISIBILITY,
        "incomplete_completion": INCOMPLETE,
    }
    panel_size = {
        "hybrid_state": 6, "erbb_dependency": 5, "akt_pathway": 6, "mtor_pathway": 6,
        "mesenchymal_state": 6, "vascular_stromal_barrier": 8,
        "immune_visibility": 8, "incomplete_completion": 8,
    }

    expression, _ = load_expression(DISCOVERY)
    z = zscore_columns(expression)
    feature = pd.read_csv(FEATURES / f"{DISCOVERY}_therapy_prioritization_features.tsv.gz", sep="\t").set_index("sample_id").loc[z.index]
    targets = target_domains(feature)
    locked = {}
    selection_rows = []
    for domain in candidates:
        selected = select_panel(z, targets[domain], candidates[domain], panel_size[domain])
        locked[domain] = [gene for gene, _ in selected]
        for rank, (gene, rho) in enumerate(selected, 1):
            selection_rows.append({
                "domain": domain, "rank": rank, "gene": gene,
                "tcga_discovery_spearman": rho, "selection_rule": "top positive gene-domain Spearman",
                "outcome_used": False,
            })
    pd.DataFrame(selection_rows).to_csv(OUT / "locked_compact_panel_genes.tsv", sep="\t", index=False)

    formula_map = {
        "hybrid_panERBB_AKT": ["hybrid_state", "erbb_dependency", "akt_pathway"],
        "hybrid_panERBB_mTORC12": ["hybrid_state", "erbb_dependency", "mtor_pathway"],
        "mesenchymal_antiangiogenic_IO": ["mesenchymal_state", "vascular_stromal_barrier", "immune_visibility", "incomplete_completion"],
    }
    validation_rows = []
    compact_scores = []
    full_scores = pd.read_csv(SCORES / "sample_level_therapy_hypothesis_scores.tsv.gz", sep="\t")
    for cohort in COHORTS:
        expression, _ = load_expression(cohort)
        z = zscore_columns(expression)
        feature = pd.read_csv(FEATURES / f"{cohort}_therapy_prioritization_features.tsv.gz", sep="\t").set_index("sample_id").loc[z.index]
        targets = target_domains(feature)
        domain_scores = pd.DataFrame(index=z.index)
        for domain, genes in locked.items():
            domain_scores[domain] = panel_domain(z, genes)
            rho = float(spearmanr(domain_scores[domain], targets[domain]).statistic)
            validation_rows.append({
                "cohort": cohort, "level": "domain", "name": domain,
                "spearman_full_vs_compact": rho, "top10_jaccard": np.nan,
            })
        out = pd.DataFrame({"sample_id": z.index, "cohort": cohort})
        cohort_full = full_scores[full_scores["cohort"] == cohort].set_index("sample_id").loc[z.index]
        for formula, domains in formula_map.items():
            compact = geomean(domain_scores[domains])
            full = cohort_full[f"score__{formula}"]
            rho = float(spearmanr(compact, full).statistic)
            compact_top = compact >= compact.quantile(0.90)
            full_top = full >= full.quantile(0.90)
            jaccard = float((compact_top & full_top).sum() / (compact_top | full_top).sum())
            validation_rows.append({
                "cohort": cohort, "level": "formula", "name": formula,
                "spearman_full_vs_compact": rho, "top10_jaccard": jaccard,
            })
            out[f"compact_score__{formula}"] = compact.to_numpy()
            out[f"compact_top10__{formula}"] = compact_top.to_numpy()
        compact_scores.append(out)

    validation = pd.DataFrame(validation_rows)
    validation["external_validation"] = validation["cohort"] != DISCOVERY
    validation.to_csv(OUT / "compact_panel_cross_cohort_validation.tsv", sep="\t", index=False)
    pd.concat(compact_scores, ignore_index=True).to_csv(
        OUT / "compact_panel_sample_scores.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    external_formula = validation[(validation["external_validation"]) & (validation["level"] == "formula")]
    passed = bool(
        (external_formula["spearman_full_vs_compact"] >= 0.60).all()
        and (external_formula["top10_jaccard"] >= 0.30).all()
    )
    summary = {
        "status": "PASS" if passed else "REVIEW",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "TCGA-derived, outcome-blind compact panel with METABRIC and SCAN-B external validation",
        "total_unique_panel_genes": len(set(sum(locked.values(), []))),
        "domains": {key: {"n": len(value), "genes": value} for key, value in locked.items()},
        "external_formula_gate": {"spearman_min": 0.60, "top10_jaccard_min": 0.30},
        "all_external_formula_gates_passed": passed,
        "outcomes_used": False,
        "treatment_response_claimed": False,
        "next_gate": "freeze assay-tier panel and prospective organoid/immune-co-culture validation matrix",
    }
    (OUT / "compact_panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
