#!/usr/bin/env python
"""Derive non-linear, redundancy-collapsed EMT-state pathway catalog."""
from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "04_analysis/11_drug_repositioning/03_state_pathway_analysis/GSE176078_malignant"
CONTINUOUS = SOURCE / "continuous_emt_pathway_models.tsv.gz"
CONTRASTS = SOURCE / "paired_state_pathway_contrasts.tsv.gz"
PROFILES = SOURCE / "patient_weighted_state_pathway_profiles.tsv.gz"
COVERAGE = SOURCE / "pathway_gene_coverage.tsv.gz"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/04_state_specific_pathway_catalog/GSE176078_malignant"

STATE_LABELS = {
    "epithelial_specific": "epithelial-enriched",
    "hybrid_specific": "hybrid-peak",
    "mesenchymal_specific": "mesenchymal-enriched",
}
CONFIDENCE_ORDER = {"high": 0, "moderate": 1, "exploratory": 2, "unresolved": 3}


def safe_logp(value: float) -> float:
    return float(-np.log10(max(float(value), 1e-300)))


def add_contrast(frame: pd.DataFrame, contrasts: pd.DataFrame, name: str, prefix: str) -> pd.DataFrame:
    columns = ["database", "pathway", "paired_patients", "mean_paired_difference", "p_value", "fdr_bh"]
    part = contrasts.loc[contrasts.contrast.eq(name), columns].copy()
    part = part.rename(columns={
        "paired_patients": f"{prefix}_n",
        "mean_paired_difference": f"{prefix}_effect",
        "p_value": f"{prefix}_p",
        "fdr_bh": f"{prefix}_fdr",
    })
    return frame.merge(part, on=["database", "pathway"], how="left", validate="one_to_one")


def classify(row) -> tuple[str, str, str]:
    # Hybrid is explicitly allowed to be a non-linear peak rather than an intermediate.
    h_signs = row.he_effect > 0 and row.mh_effect < 0
    h_profile = row.hybrid_score > row.epithelial_score and row.hybrid_score > row.mesenchymal_score
    h_axis = row.coactivation_beta > 0 and row.coactivation_fdr < 0.05
    if h_axis and h_signs and h_profile:
        confidence = "high" if row.he_p < 0.05 and row.mh_p < 0.05 else "moderate"
        return "hybrid_specific", confidence, "coactivation-positive with hybrid above both flanking states"

    # E/M specificity uses the well-powered continuous position term plus paired direction.
    e_signs = row.he_effect < 0 and row.me_effect < 0  # H-E <0 and M-E <0 => E is higher.
    e_profile = row.epithelial_score > row.hybrid_score and row.epithelial_score > row.mesenchymal_score
    e_axis = row.position_beta < 0 and row.position_fdr < 0.05
    if e_axis and e_signs and e_profile:
        confidence = "high" if row.he_p < 0.05 else "moderate"
        return "epithelial_specific", confidence, "negative EMT-position association with epithelial profile maximum"

    m_signs = row.mh_effect > 0 and row.me_effect > 0  # M-H >0 and M-E >0 => M is higher.
    m_profile = row.mesenchymal_score > row.hybrid_score and row.mesenchymal_score > row.epithelial_score
    m_axis = row.position_beta > 0 and row.position_fdr < 0.05
    if m_axis and m_signs and m_profile:
        confidence = "high" if row.mh_p < 0.05 else "moderate"
        return "mesenchymal_specific", confidence, "positive EMT-position association with mesenchymal profile maximum"

    # Exploratory state maxima are retained for bulk replication but cannot drive a drug query.
    scores = {
        "epithelial_specific": row.epithelial_score,
        "hybrid_specific": row.hybrid_score,
        "mesenchymal_specific": row.mesenchymal_score,
    }
    best = max(scores, key=scores.get)
    return best, "exploratory", "descriptive state maximum without complete continuous/paired support"


def evidence_score(row) -> float:
    if row.state_class == "hybrid_specific":
        axis_fdr = row.coactivation_fdr
        paired_p = max(row.he_p, row.mh_p)
        margin = row.hybrid_score - max(row.epithelial_score, row.mesenchymal_score)
    elif row.state_class == "epithelial_specific":
        axis_fdr = row.position_fdr
        paired_p = row.he_p
        margin = row.epithelial_score - max(row.hybrid_score, row.mesenchymal_score)
    else:
        axis_fdr = row.position_fdr
        paired_p = row.mh_p
        margin = row.mesenchymal_score - max(row.hybrid_score, row.epithelial_score)
    return safe_logp(axis_fdr) + safe_logp(paired_p) + max(float(margin), 0.0)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collapse_redundancy(group: pd.DataFrame) -> pd.DataFrame:
    group = group.reset_index(drop=True).copy()
    sets = [set(str(value).split(";")) - {""} for value in group.detected_gene_symbols]
    uf = UnionFind(len(group))
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            intersection = len(sets[i] & sets[j])
            if intersection == 0:
                continue
            overlap_coefficient = intersection / min(len(sets[i]), len(sets[j]))
            jaccard = intersection / len(sets[i] | sets[j])
            if overlap_coefficient >= 0.70 or jaccard >= 0.50:
                uf.union(i, j)
    group["redundancy_cluster"] = [uf.find(i) for i in range(len(group))]
    cluster_map = {old: new for new, old in enumerate(sorted(group.redundancy_cluster.unique()), start=1)}
    group["redundancy_cluster"] = group.redundancy_cluster.map(cluster_map)
    group["is_representative"] = False
    for _, indices in group.groupby("redundancy_cluster").groups.items():
        candidates = group.loc[list(indices)].copy()
        candidates["confidence_rank"] = candidates.confidence.map(CONFIDENCE_ORDER)
        selected = candidates.sort_values(
            ["confidence_rank", "evidence_score", "detected_genes"],
            ascending=[True, False, True],
        ).index[0]
        group.loc[selected, "is_representative"] = True
    return group


def main() -> None:
    for path in [CONTINUOUS, CONTRASTS, PROFILES, COVERAGE, LOCK]:
        if not path.exists():
            raise FileNotFoundError(path)
    continuous = pd.read_csv(CONTINUOUS, sep="\t")
    contrasts = pd.read_csv(CONTRASTS, sep="\t")
    profiles = pd.read_csv(PROFILES, sep="\t")
    coverage = pd.read_csv(COVERAGE, sep="\t")

    position = continuous.loc[continuous.term.eq("emt_position_within_patient"),
                              ["database", "pathway", "beta", "p_value", "fdr_bh"]].rename(columns={
                                  "beta": "position_beta", "p_value": "position_p", "fdr_bh": "position_fdr"})
    coactivation = continuous.loc[continuous.term.eq("coactivation_within_patient"),
                                  ["database", "pathway", "beta", "p_value", "fdr_bh"]].rename(columns={
                                      "beta": "coactivation_beta", "p_value": "coactivation_p", "fdr_bh": "coactivation_fdr"})
    catalog = coverage.merge(position, on=["database", "pathway"], validate="one_to_one")
    catalog = catalog.merge(coactivation, on=["database", "pathway"], validate="one_to_one")
    catalog = add_contrast(catalog, contrasts, "hybrid_minus_epithelial", "he")
    catalog = add_contrast(catalog, contrasts, "mesenchymal_minus_hybrid", "mh")
    catalog = add_contrast(catalog, contrasts, "mesenchymal_minus_epithelial", "me")

    profile_wide = profiles.pivot(index=["database", "pathway"], columns="emt_state", values="mean_score").reset_index()
    profile_wide = profile_wide.rename(columns={
        "epithelial_dominant": "epithelial_score",
        "hybrid_EM": "hybrid_score",
        "mesenchymal_dominant": "mesenchymal_score",
    })
    catalog = catalog.merge(profile_wide, on=["database", "pathway"], validate="one_to_one")
    classified = catalog.apply(classify, axis=1, result_type="expand")
    classified.columns = ["state_class", "confidence", "classification_basis"]
    catalog = pd.concat([catalog, classified], axis=1)
    catalog["state_label"] = catalog.state_class.map(STATE_LABELS)
    catalog["evidence_score"] = catalog.apply(evidence_score, axis=1)

    supported = catalog.loc[catalog.confidence.isin(["high", "moderate"])].copy()
    collapsed_parts = []
    for (state_class, database), group in supported.groupby(["state_class", "database"], sort=True):
        collapsed_parts.append(collapse_redundancy(group))
    collapsed = pd.concat(collapsed_parts, ignore_index=True) if collapsed_parts else pd.DataFrame()
    representatives = collapsed.loc[collapsed.is_representative].copy() if len(collapsed) else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT / "all_pathway_state_classification.tsv.gz", sep="\t", index=False, compression="gzip")
    collapsed.to_csv(OUT / "supported_state_pathways_redundancy_clusters.tsv", sep="\t", index=False)
    representatives.to_csv(OUT / "state_specific_pathway_representatives.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "analysis_type": "non-linear EMT-state pathway specificity with redundancy collapse",
        "pathways_classified": int(len(catalog)),
        "supported_before_collapse": {
            state: int(len(group)) for state, group in supported.groupby("state_label")
        },
        "representatives_after_collapse": {
            state: int(len(group)) for state, group in representatives.groupby("state_label")
        } if len(representatives) else {},
        "high_confidence": {
            state: int(len(group)) for state, group in supported.loc[supported.confidence.eq("high")].groupby("state_label")
        },
        "interpretation": {
            "hybrid": "non-linear peak allowed and tested against both epithelial and mesenchymal states",
            "epithelial_mesenchymal": "continuous EMT-position evidence required; four-patient direct E-M contrast is directional only",
            "pathway_specific": "relative state enrichment, not absolute expression restricted to one state",
        },
        "next_gate": "independent bulk-cohort replication of representative pathway directions before state-specific drug mapping",
    }
    (OUT / "state_specific_pathway_catalog_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(representatives):
        print("\nTop state-specific representatives:")
        print(representatives.sort_values(["state_label", "confidence", "evidence_score"],
              ascending=[True, True, False])[["state_label", "confidence", "database", "pathway", "evidence_score"]]
              .groupby("state_label").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
