#!/usr/bin/env python
"""Collapse redundant bulk-replicated pathways and assign provisional biological roles."""
from __future__ import annotations

import os

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CATALOG = ROOT / "04_analysis/11_drug_repositioning/05_bulk_state_pathway_replication/multiscale_state_specific_pathway_catalog.tsv"
COVERAGE = ROOT / "04_analysis/11_drug_repositioning/03_state_pathway_analysis/GSE176078_malignant/pathway_gene_coverage.tsv.gz"
OUT = ROOT / "04_analysis/11_drug_repositioning/06_collapsed_state_pathway_map"

CONFIDENCE_ORDER = {"very_high": 0, "high": 1, "moderate": 2, "exploratory": 3}

THEMES = [
    ("immune_inflammatory", r"interferon|interleukin|chemokine|cytokine|complement|toll|tlr|dap12|immune|antigen|mhc|pd-?1|nf.?kb|allograft|inflam|leukocyte|lymph|macrophage|neutrophil"),
    ("adhesion_ecm_motility", r"extracellular matrix|\becm\b|integrin|collagen|elastic|adhesion|junction|hemidesmosome|semaphorin|ephrin|epha|ephb|rho |rho-|rho_|iqgap|motility|migration|cytoskeleton|focal adhesion|gap junction|basigin|ncam"),
    ("growth_survival_signaling", r"erbb|egfr|pi3k|akt|mtor|mapk|ras|raf|pdgf|fgf|notch|wnt|hedgehog|jak|stat|tgf|smad|receptor tyrosine|growth factor|ptk6"),
    ("hypoxia_angiogenesis", r"hypoxia|angiogen|vegf|vascular|endothelial"),
    ("metabolic_adaptation", r"metabol|oxidative phosphorylation|respiration|mitochond|peroxisom|fatty acid|lipid|cholesterol|bile acid|amino acid|glycolysis|gluconeogenesis|pyruvate|tca|carnitine|glycogen|folate|vitamin|cofactor|nucleotide|steroid|glycos|hexose|adipogenesis"),
    ("protein_rna_homeostasis", r"protein folding|unfolded protein|calnexin|calreticulin|er quality|proteasom|autophagy|lysosom|golgi|vesicle|trafficking|transport|translation|ribosom|rna polymerase|transcription|splicing|mrna|snrna|trna|protein localization|protein import|protein degradation|secretion"),
    ("cell_cycle_dna_damage", r"cell cycle|mitotic|mitosis|dna repair|replication|chromatin|checkpoint|apoptosis|p53|senescence"),
    ("coagulation_platelet", r"coagulation|clot|platelet|hemostasis|fibrin"),
]

ROLE_MAP = {
    "immune_inflammatory": "immune_context_or_bottleneck",
    "adhesion_ecm_motility": "barrier_plasticity_candidate",
    "growth_survival_signaling": "signaling_driver_candidate",
    "hypoxia_angiogenesis": "microenvironmental_bottleneck_candidate",
    "metabolic_adaptation": "metabolic_dependency_or_adaptation",
    "protein_rna_homeostasis": "stress_adaptation_candidate",
    "cell_cycle_dna_damage": "proliferation_or_damage_response",
    "coagulation_platelet": "vascular_stromal_context_candidate",
    "other": "marker_or_unresolved",
}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def theme(pathway: str) -> str:
    text = pathway.lower()
    for label, pattern in THEMES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "other"


def defining_evidence(row) -> float:
    if row.bulk_state_class == "epithelial-enriched":
        fdrs = [row.he_pooled_fdr_bh, row.me_pooled_fdr_bh]
    elif row.bulk_state_class == "hybrid-peak":
        fdrs = [row.he_pooled_fdr_bh, row.mh_pooled_fdr_bh]
    else:
        fdrs = [row.mh_pooled_fdr_bh, row.me_pooled_fdr_bh]
    return float(sum(-np.log10(max(float(value), 1e-300)) for value in fdrs))


def collapse(group: pd.DataFrame) -> pd.DataFrame:
    group = group.reset_index(drop=True).copy()
    genes = [set(str(value).split(";")) - {""} for value in group.detected_gene_symbols]
    uf = UnionFind(len(group))
    for left in range(len(group)):
        for right in range(left + 1, len(group)):
            overlap = len(genes[left] & genes[right])
            if not overlap:
                continue
            coefficient = overlap / min(len(genes[left]), len(genes[right]))
            jaccard = overlap / len(genes[left] | genes[right])
            if coefficient >= 0.70 or jaccard >= 0.50:
                uf.union(left, right)
    roots = [uf.find(index) for index in range(len(group))]
    mapping = {root: rank for rank, root in enumerate(sorted(set(roots)), start=1)}
    group["redundancy_cluster"] = [mapping[root] for root in roots]
    group["cluster_size"] = group.groupby("redundancy_cluster").pathway.transform("size")
    group["representative"] = False
    for _, indices in group.groupby("redundancy_cluster").groups.items():
        candidates = group.loc[list(indices)].copy()
        candidates["confidence_rank"] = candidates.multiscale_confidence.map(CONFIDENCE_ORDER)
        selected = candidates.sort_values(
            ["confidence_rank", "defining_evidence_score", "detected_genes"],
            ascending=[True, False, True],
        ).index[0]
        group.loc[selected, "representative"] = True
    return group


def main() -> None:
    catalog = pd.read_csv(CATALOG, sep="\t")
    coverage = pd.read_csv(COVERAGE, sep="\t")
    supported = catalog.loc[catalog.multiscale_confidence.isin(["very_high", "high", "moderate"])].copy()
    supported = supported.merge(
        coverage[["database", "pathway", "detected_genes", "detected_gene_symbols"]],
        on=["database", "pathway"], how="left", validate="one_to_one"
    )
    if supported.detected_gene_symbols.isna().any():
        raise ValueError("Missing pathway gene membership")
    supported["biological_theme"] = supported.pathway.map(theme)
    supported["provisional_intervention_role"] = supported.biological_theme.map(ROLE_MAP)
    supported["cell_origin_evidence"] = np.where(
        supported.single_cell_concordant & supported.single_cell_confidence.isin(["high", "moderate"]),
        "malignant_supported", "bulk_ecosystem_cell_origin_unresolved"
    )
    supported["defining_evidence_score"] = supported.apply(defining_evidence, axis=1)

    parts = []
    for (state, database), group in supported.groupby(["bulk_state_class", "database"], sort=True):
        parts.append(collapse(group))
    collapsed = pd.concat(parts, ignore_index=True)
    representatives = collapsed.loc[collapsed.representative].copy()
    representatives = representatives.sort_values(
        ["bulk_state_class", "multiscale_confidence", "defining_evidence_score"],
        key=lambda column: column.map(CONFIDENCE_ORDER) if column.name == "multiscale_confidence" else column,
        ascending=[True, True, False],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(OUT / "supported_pathway_redundancy_clusters.tsv.gz", sep="\t", index=False, compression="gzip")
    representatives.to_csv(OUT / "state_pathway_representatives_annotated.tsv", sep="\t", index=False)
    theme_counts = representatives.groupby([
        "bulk_state_class", "biological_theme", "provisional_intervention_role"
    ]).size().reset_index(name="representative_pathways")
    theme_counts.to_csv(OUT / "state_biological_theme_counts.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "analysis_type": "redundancy-collapsed multiscale state pathway map with provisional roles",
        "supported_before_collapse": int(len(supported)),
        "representatives_after_collapse": int(len(representatives)),
        "representatives_by_state": {
            state: int(len(group)) for state, group in representatives.groupby("bulk_state_class")
        },
        "representatives_by_state_and_theme": {
            state: {theme_name: int(len(theme_group)) for theme_name, theme_group in state_group.groupby("biological_theme")}
            for state, state_group in representatives.groupby("bulk_state_class")
        },
        "malignant_supported_representatives": int(representatives.cell_origin_evidence.eq("malignant_supported").sum()),
        "interpretation_constraint": "roles are provisional; driver status requires directed-network, compartment and perturbation evidence",
        "next_gate": "OmniPath direction/compartment mapping and ChEMBL/LINCS targetability annotation",
    }
    (OUT / "collapsed_state_pathway_map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop representatives by state:")
    print(representatives.groupby("bulk_state_class").head(15)[
        ["bulk_state_class", "multiscale_confidence", "biological_theme", "pathway", "cell_origin_evidence"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
