#!/usr/bin/env python
"""Replicate state-specific pathway activity across TCGA, METABRIC and SCAN-B."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, t as t_distribution


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXPRESSION = ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival"
LANDSCAPE = ROOT / "04_analysis/08_biological_landscape"
STATE_FILE = LANDSCAPE / "emt_state_immune_archetypes/sample_emt_and_immune_states.tsv.gz"
PATHWAYS = ROOT / "04_analysis/11_drug_repositioning/03_state_pathway_analysis/GSE176078_malignant/pathway_gene_coverage.tsv.gz"
SC_CATALOG = ROOT / "04_analysis/11_drug_repositioning/04_state_specific_pathway_catalog/GSE176078_malignant/all_pathway_state_classification.tsv.gz"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/05_bulk_state_pathway_replication"

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

STATE_LEVELS = ["epithelial_dominant", "hybrid_EM", "mesenchymal_dominant", "double_low_indeterminate"]
CONTRASTS = {
    "hybrid_minus_epithelial": ("hybrid_EM", "epithelial_dominant"),
    "mesenchymal_minus_hybrid": ("mesenchymal_dominant", "hybrid_EM"),
    "mesenchymal_minus_epithelial": ("mesenchymal_dominant", "epithelial_dominant"),
}


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    result = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return result
    x = p[good]
    order = np.argsort(x)
    adjusted = x[order] * len(x) / np.arange(1, len(x) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[good] = restored
    return result


def load_expression(path: Path):
    with h5py.File(path, "r") as handle:
        x = handle["X"][...].astype(np.float32)
        samples = decode(handle["sample_id"][...])
        genes = [value.upper() for value in decode(handle["gene_symbol"][...])]
    return x, samples, genes


def build_membership(pathways: pd.DataFrame, genes: list[str]):
    lookup = {gene: index for index, gene in enumerate(genes)}
    rows, columns, data, retained = [], [], [], []
    for item in pathways.itertuples():
        source = [gene for gene in str(item.detected_gene_symbols).split(";") if gene]
        measured = [gene for gene in source if gene in lookup]
        if len(measured) < 10 or len(measured) / len(source) < 0.70:
            continue
        column = len(retained)
        retained.append({
            "database": item.database, "pathway": item.pathway,
            "single_cell_detected_genes": len(source), "bulk_measured_genes": len(measured),
            "bulk_coverage": len(measured) / len(source),
        })
        weight = 1.0 / len(measured)
        for gene in measured:
            rows.append(lookup[gene]); columns.append(column); data.append(weight)
    matrix = sparse.csc_matrix((data, (rows, columns)), shape=(len(genes), len(retained)), dtype=np.float32)
    return matrix, pd.DataFrame(retained)


def design_matrix(metadata: pd.DataFrame):
    state = pd.get_dummies(pd.Categorical(metadata.emt_state, categories=STATE_LEVELS), dtype=float)
    state.columns = [f"state::{column}" for column in state.columns]
    pam50 = pd.get_dummies(metadata.pam50.fillna("Unknown").astype(str), prefix="pam50", drop_first=True, dtype=float)
    stromal = pd.to_numeric(metadata.stromal_z, errors="raise").rename("stromal_z")
    design = pd.concat([state, pam50, stromal], axis=1)
    return design


def fit_all(scores: np.ndarray, metadata: pd.DataFrame, manifest: pd.DataFrame, cohort: str):
    design = design_matrix(metadata)
    x = design.to_numpy(float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ scores
    residual = scores - x @ beta
    df = len(metadata) - x.shape[1]
    sigma2 = np.sum(residual ** 2, axis=0) / df
    names = list(design.columns)
    rows = []
    for contrast, (numerator, denominator) in CONTRASTS.items():
        vector = np.zeros(x.shape[1])
        vector[names.index(f"state::{numerator}")] = 1.0
        vector[names.index(f"state::{denominator}")] = -1.0
        effect = vector @ beta
        factor = float(vector @ bread @ vector)
        se = np.sqrt(np.maximum(sigma2 * factor, 0))
        statistic = np.divide(effect, se, out=np.zeros_like(effect), where=se > 0)
        p = 2 * t_distribution.sf(np.abs(statistic), df=df)
        block = manifest.copy()
        block["cohort"] = cohort
        block["contrast"] = contrast
        block["n"] = len(metadata)
        block["adjusted_difference"] = effect
        block["standard_error"] = se
        block["t_value"] = statistic
        block["p_value"] = p
        block["fdr_bh"] = block.groupby("database", group_keys=False).p_value.transform(bh)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def meta_analyze(cohort_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (database, pathway, contrast), group in cohort_results.groupby(["database", "pathway", "contrast"], sort=False):
        effect = group.adjusted_difference.to_numpy(float)
        se = group.standard_error.to_numpy(float)
        weights = 1.0 / np.square(se)
        pooled = float(np.sum(weights * effect) / np.sum(weights))
        pooled_se = float(1.0 / np.sqrt(np.sum(weights)))
        z = pooled / pooled_se
        rows.append({
            "database": database, "pathway": pathway, "contrast": contrast,
            "cohorts": len(group), "pooled_adjusted_difference": pooled,
            "pooled_standard_error": pooled_se, "pooled_z": z,
            "pooled_p_value": float(2 * norm.sf(abs(z))),
            "same_direction_all_cohorts": bool(np.all(np.sign(effect) == np.sign(effect[0]))),
            "TCGA_effect": float(group.loc[group.cohort.eq("TCGA_BRCA"), "adjusted_difference"].iloc[0]),
            "METABRIC_effect": float(group.loc[group.cohort.eq("METABRIC"), "adjusted_difference"].iloc[0]),
            "SCANB_effect": float(group.loc[group.cohort.eq("SCANB_GSE96058"), "adjusted_difference"].iloc[0]),
        })
    meta = pd.DataFrame(rows)
    meta["pooled_fdr_bh"] = meta.groupby(["database", "contrast"], group_keys=False).pooled_p_value.transform(bh)
    return meta


def classify(meta: pd.DataFrame, sc_catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["database", "pathway", "contrast", "pooled_adjusted_difference", "pooled_p_value",
               "pooled_fdr_bh", "same_direction_all_cohorts", "TCGA_effect", "METABRIC_effect", "SCANB_effect"]
    wide = None
    for contrast in CONTRASTS:
        part = meta.loc[meta.contrast.eq(contrast), columns].copy()
        prefix = {"hybrid_minus_epithelial": "he", "mesenchymal_minus_hybrid": "mh",
                  "mesenchymal_minus_epithelial": "me"}[contrast]
        part = part.drop(columns="contrast").rename(columns={
            column: f"{prefix}_{column}" for column in part.columns if column not in ["database", "pathway"]
        })
        wide = part if wide is None else wide.merge(part, on=["database", "pathway"], validate="one_to_one")

    classifications = []
    for row in wide.itertuples():
        rules = {
            "epithelial-enriched": row.he_pooled_adjusted_difference < 0 and row.me_pooled_adjusted_difference < 0,
            "hybrid-peak": row.he_pooled_adjusted_difference > 0 and row.mh_pooled_adjusted_difference < 0,
            "mesenchymal-enriched": row.mh_pooled_adjusted_difference > 0 and row.me_pooled_adjusted_difference > 0,
        }
        state = next((name for name, passed in rules.items() if passed), "non_specific")
        if state == "epithelial-enriched":
            prefixes = ["he", "me"]
        elif state == "hybrid-peak":
            prefixes = ["he", "mh"]
        elif state == "mesenchymal-enriched":
            prefixes = ["mh", "me"]
        else:
            prefixes = []
        fdrs = [getattr(row, f"{prefix}_pooled_fdr_bh") for prefix in prefixes]
        replicated = [getattr(row, f"{prefix}_same_direction_all_cohorts") for prefix in prefixes]
        pvalues = [getattr(row, f"{prefix}_pooled_p_value") for prefix in prefixes]
        if prefixes and all(value < 0.05 for value in fdrs) and all(replicated):
            confidence = "high"
        elif prefixes and all(value < 0.05 for value in pvalues) and all(replicated):
            confidence = "moderate"
        else:
            confidence = "exploratory" if prefixes else "unresolved"
        classifications.append((state, confidence))
    wide[["bulk_state_class", "bulk_confidence"]] = pd.DataFrame(classifications, index=wide.index)

    sc = sc_catalog[["database", "pathway", "state_label", "confidence"]].rename(columns={
        "state_label": "single_cell_state_class", "confidence": "single_cell_confidence"})
    wide = wide.merge(sc, on=["database", "pathway"], how="left", validate="one_to_one")
    wide["single_cell_concordant"] = wide.bulk_state_class.eq(wide.single_cell_state_class)
    wide["multiscale_confidence"] = np.select([
        wide.bulk_confidence.eq("high") & wide.single_cell_concordant & wide.single_cell_confidence.isin(["high", "moderate"]),
        wide.bulk_confidence.eq("high"),
        wide.bulk_confidence.eq("moderate") & wide.single_cell_concordant,
    ], ["very_high", "high", "moderate"], default="exploratory")
    return wide


def main() -> None:
    for path in [STATE_FILE, PATHWAYS, SC_CATALOG, LOCK, *[x for cfg in CONFIG.values() for x in cfg.values()]]:
        if not path.exists():
            raise FileNotFoundError(path)
    states = pd.read_csv(STATE_FILE, sep="\t", dtype={"sample_id": str})[["sample_id", "cohort", "emt_state"]]
    pathways = pd.read_csv(PATHWAYS, sep="\t")
    sc_catalog = pd.read_csv(SC_CATALOG, sep="\t")
    all_results, cohort_summary = [], {}

    for cohort, cfg in CONFIG.items():
        expression, sample_ids, genes = load_expression(cfg["expression"])
        axes = pd.read_csv(cfg["axes"], sep="\t", dtype={"sample_id": str})[["sample_id", "pam50", "stromal_z"]]
        metadata = pd.DataFrame({"sample_id": sample_ids}).merge(
            states.loc[states.cohort.eq(cohort), ["sample_id", "emt_state"]], on="sample_id", validate="one_to_one"
        ).merge(axes, on="sample_id", validate="one_to_one")
        if len(metadata) != len(sample_ids):
            raise ValueError(f"{cohort}: incomplete metadata linkage {len(metadata)}/{len(sample_ids)}")
        membership, manifest = build_membership(pathways, genes)
        means = expression.mean(axis=0)
        sd = expression.std(axis=0, ddof=0)
        variable = sd > 0
        expression[:, variable] = (expression[:, variable] - means[variable]) / sd[variable]
        expression[:, ~variable] = 0
        scores = np.asarray(expression @ membership, dtype=np.float32)
        results = fit_all(scores, metadata, manifest, cohort)
        all_results.append(results)
        cohort_summary[cohort] = {
            "samples": len(metadata), "pathways_scored": len(manifest),
            "emt_state_counts": {str(k): int(v) for k, v in metadata.emt_state.value_counts().items()},
        }
        del expression, scores

    cohort_results = pd.concat(all_results, ignore_index=True)
    meta = meta_analyze(cohort_results)
    catalog = classify(meta, sc_catalog)
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_results.to_csv(OUT / "cohort_adjusted_state_pathway_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")
    meta.to_csv(OUT / "three_cohort_state_pathway_meta_analysis.tsv.gz", sep="\t", index=False, compression="gzip")
    catalog.to_csv(OUT / "multiscale_state_specific_pathway_catalog.tsv", sep="\t", index=False)

    supported = catalog.loc[catalog.multiscale_confidence.isin(["very_high", "high", "moderate"])]
    summary = {
        "status": "PASS",
        "analysis_type": "outcome-free three-cohort adjusted replication of EMT-state pathways",
        "cohorts": cohort_summary,
        "total_samples": int(sum(item["samples"] for item in cohort_summary.values())),
        "pathways_evaluated": int(len(catalog)),
        "supported_by_state": {
            state: int(len(group)) for state, group in supported.groupby("bulk_state_class")
        },
        "very_high_multiscale_by_state": {
            state: int(len(group)) for state, group in catalog.loc[catalog.multiscale_confidence.eq("very_high")].groupby("bulk_state_class")
        },
        "interpretation": "state specificity requires the two defining contrasts; high confidence also requires same direction in all three bulk cohorts",
        "next_gate": "redundancy collapse and functional driver/bottleneck/marker annotation before drug mapping",
    }
    (OUT / "bulk_state_pathway_replication_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
