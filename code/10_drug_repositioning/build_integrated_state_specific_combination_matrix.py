#!/usr/bin/env python
"""Integrate pathway, LINCS, and pharmacogenomic evidence into state-specific combinations.

This is an evidence-synthesis step, not a synergy test.  It deliberately separates
validated anchors from mechanistic partners and prevents single-screen or
covariate-dependent hits from being promoted as final therapeutic candidates.
"""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
PATHWAYS = BASE / "06_collapsed_state_pathway_map/state_pathway_representatives_annotated.tsv"
TARGETS = BASE / "07_state_druggable_target_map/state_druggable_target_evidence.tsv"
MECHANISMS = BASE / "07_state_druggable_target_map/state_chembl_drug_mechanism_candidates.tsv.gz"
LINCS = BASE / "09_state_specific_lincs_scoring/lincs_mechanism_supported_candidates_24h.tsv"
PGX = BASE / "11_pharmacogenomic_validation/state_specific_pharmacogenomic_synthesis.tsv"
TARGET_FIRST = BASE / "13_target_first_dependency_screen/target_first_state_dependency_synthesis.tsv"
TESTS = BASE / "13_target_first_dependency_screen/target_first_state_dependency_tests.tsv.gz"
OUT = BASE / "14_integrated_state_specific_combination_matrix"


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def require_row(frame: pd.DataFrame, state: str, drug: str, drug_col: str) -> pd.Series:
    hit = frame[
        frame["state"].eq(state)
        & frame[drug_col].astype(str).str.casefold().eq(drug.casefold())
    ]
    if hit.empty:
        raise ValueError(f"Required frozen evidence missing: state={state}, drug={drug}, source={drug_col}")
    return hit.iloc[0]


def class_support(target_first: pd.DataFrame, state: str, genes: set[str]) -> dict[str, object]:
    subset = target_first[target_first["state"].eq(state)].copy()
    subset["gene_set"] = subset["state_target_genes"].fillna("").map(lambda x: set(str(x).split(";")))
    subset = subset[subset["gene_set"].map(lambda x: bool(x & genes))]
    reproducible = subset[
        subset["dependency_evidence_class"].isin(
            [
                "global_fdr_and_cross_screen_direction",
                "screen_fdr_and_cross_screen_direction",
                "replicated_direction_with_strong_effect",
            ]
        )
    ]
    return {
        "class_drugs_tested": int(len(subset)),
        "class_reproducible_drugs": int(len(reproducible)),
        "class_reproducible_names": ";".join(sorted(reproducible["drug"].astype(str).unique())),
    }


def mechanism_available(mechanisms: pd.DataFrame, state: str, gene: str, molecule: str) -> bool:
    mask = (
        mechanisms["state"].eq(state)
        & mechanisms["gene"].eq(gene)
        & mechanisms["molecule_name"].astype(str).str.casefold().eq(molecule.casefold())
        & mechanisms["mechanism_direction_compatible"].astype(str).str.casefold().eq("true")
    )
    return bool(mask.any())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pathways = pd.read_csv(PATHWAYS, sep="\t", low_memory=False)
    targets = pd.read_csv(TARGETS, sep="\t", low_memory=False)
    mechanisms = pd.read_csv(MECHANISMS, sep="\t", low_memory=False)
    lincs = pd.read_csv(LINCS, sep="\t", low_memory=False)
    pgx = pd.read_csv(PGX, sep="\t", low_memory=False)
    target_first = pd.read_csv(TARGET_FIRST, sep="\t", low_memory=False)
    tests = pd.read_csv(TESTS, sep="\t", low_memory=False)

    target_first = numeric(target_first, ["best_global_fdr", "best_cliffs_delta", "screens_tested"])
    tests = numeric(
        tests,
        [
            "matched_minus_other_median",
            "cliffs_delta",
            "subtype_adjusted_state_beta",
            "subtype_adjusted_one_sided_p",
            "primary_fdr_bh_global",
        ],
    )

    # Guardrails: the integrated conclusions below are only valid if the frozen
    # evidence that motivated them is present and has not silently changed.
    bms = require_row(target_first, "hybrid-peak", "BMS-690514", "drug")
    pozi = require_row(target_first, "hybrid-peak", "poziotinib", "drug")
    briv = require_row(pgx, "mesenchymal-enriched", "brivanib", "candidate")
    require_row(target_first, "epithelial-enriched", "ODM-201", "drug")
    require_row(target_first, "epithelial-enriched", "triapine", "drug")

    if float(bms["best_global_fdr"]) >= 0.10:
        raise ValueError("BMS-690514 no longer passes the frozen global-FDR anchor gate")
    if int(pozi["positive_subtype_adjusted_screens"]) < 1:
        raise ValueError("Poziotinib no longer supports the ERBB2/ERBB3 class direction")
    if str(briv["direction_compatible_chembl_match"]).casefold() != "true":
        raise ValueError("Brivanib no longer has direction-compatible mechanism support")

    h_class = class_support(target_first, "hybrid-peak", {"ERBB2", "ERBB3"})
    if h_class["class_reproducible_drugs"] < 3:
        raise ValueError("Hybrid ERBB-family conclusion lacks the required class-level replication")

    bms_tests = tests[
        tests["state"].eq("hybrid-peak")
        & tests["drug"].astype(str).str.casefold().eq("bms-690514")
    ]
    bms_primary = bms_tests[bms_tests["screen"].eq("PRISM_primary")].iloc[0]

    epithelial_issues = (
        "ODM-201: global FDR but small raw Cliff delta and single screen; "
        "triapine: subtype-adjusted signal opposes the raw median/Cliff direction"
    )

    anchors = pd.DataFrame(
        [
            {
                "state": "epithelial-enriched",
                "anchor_or_class": "none promoted",
                "target_axis": "MDM2 or PI3K-AKT-mTOR remain exploratory",
                "evidence_status": "unresolved",
                "best_global_fdr": np.nan,
                "best_effect": np.nan,
                "cross_screen_or_class_replication": False,
                "compartment": "malignant/shared",
                "decision": "do_not_promote_a_combination_yet",
                "constraint": epithelial_issues,
            },
            {
                "state": "hybrid-peak",
                "anchor_or_class": "BMS-690514; class support: poziotinib/AZD8931 and other pan-ErbB inhibitors",
                "target_axis": "ERBB2/ERBB3",
                "evidence_status": "validated_state_dependency_anchor",
                "best_global_fdr": float(bms["best_global_fdr"]),
                "best_effect": float(bms_primary["cliffs_delta"]),
                "cross_screen_or_class_replication": True,
                "compartment": "malignant",
                "decision": "promote_anchor_to_combination_prioritization",
                "constraint": "five hybrid models; validate biomarker-defined effect in independent models",
            },
            {
                "state": "mesenchymal-enriched",
                "anchor_or_class": "brivanib",
                "target_axis": "FGFR1/VEGFR2(KDR)/VEGFR3(FLT4)",
                "evidence_status": "convergent_nominal_remodelling_anchor",
                "best_global_fdr": np.nan,
                "best_effect": 0.472,
                "cross_screen_or_class_replication": False,
                "compartment": "stromal_vascular",
                "decision": "retain_as_hypothesis_not_validated_biomarker",
                "constraint": "LINCS reversal plus one-screen nominal viability; no global FDR support",
            },
        ]
    )
    for key, value in h_class.items():
        anchors.loc[anchors["state"].eq("hybrid-peak"), key] = value

    # These are mechanistically complementary hypotheses, not observed synergy.
    combinations = pd.DataFrame(
        [
            {
                "priority": 1,
                "state": "hybrid-peak",
                "anchor": "pan-ErbB inhibitor class (BMS-690514 discovery anchor; poziotinib/AZD8931 class support)",
                "partner": "ROCK1/ROCK2 inhibitor class (belumosudil or fasudil as repositioning leads)",
                "anchor_targets": "ERBB2;ERBB3",
                "partner_targets": "ROCK1;ROCK2",
                "biological_objective": "hit hybrid-state growth wiring while disrupting RHO/adhesion/motility and stromal-barrier behavior",
                "immune_logic": "preserve the already visible immune compartment; test whether barrier reduction enables effector completion",
                "evidence_tier": "Tier_1_anchor_plus_Tier_3_partner",
                "synergy_demonstrated": False,
                "required_next_test": "6x6 dose matrix in hybrid-high models plus epithelial/mesenchymal controls; Bliss/ZIP and immune-preservation readouts",
            },
            {
                "priority": 2,
                "state": "hybrid-peak",
                "anchor": "pan-ErbB inhibitor class",
                "partner": "PD-1/PD-L1 checkpoint-release class",
                "anchor_targets": "ERBB2;ERBB3",
                "partner_targets": "PDCD1;CD274",
                "biological_objective": "couple a malignant hybrid dependency to immune completion",
                "immune_logic": "hybrid tumors are relatively immune-visible but incompletely executing; checkpoint release is conditional on preserved T-cell competence",
                "evidence_tier": "Tier_1_anchor_plus_mechanistic_immune_partner",
                "synergy_demonstrated": False,
                "required_next_test": "tumor-immune co-culture or organoid assay; cytotoxicity, IFNG/GZMB, antigen presentation and T-cell viability",
            },
            {
                "priority": 1,
                "state": "mesenchymal-enriched",
                "anchor": "brivanib",
                "partner": "PD-1/PD-L1 checkpoint-release class",
                "anchor_targets": "FGFR1;KDR;FLT4",
                "partner_targets": "PDCD1;CD274",
                "biological_objective": "remodel angiogenic/stromal exclusion while releasing the engaged-but-incomplete immune response",
                "immune_logic": "anti-angiogenic/FGFR remodeling is intended to improve access, not simply kill mesenchymal cells",
                "evidence_tier": "Tier_2_remodelling_anchor_plus_mechanistic_immune_partner",
                "synergy_demonstrated": False,
                "required_next_test": "endothelial-stromal-tumor-immune co-culture; perfusion/barrier, T-cell infiltration and killing endpoints",
            },
            {
                "priority": 2,
                "state": "mesenchymal-enriched",
                "anchor": "brivanib",
                "partner": "selective COX-2 inhibition (celecoxib repositioning lead)",
                "anchor_targets": "FGFR1;KDR;FLT4",
                "partner_targets": "PTGS2",
                "biological_objective": "co-target angiogenic/ECM remodeling and inflammatory immune suppression",
                "immune_logic": "requires direct demonstration that PTGS2-high mesenchymal niches suppress rather than support productive immunity",
                "evidence_tier": "Tier_2_anchor_plus_Tier_3_partner",
                "synergy_demonstrated": False,
                "required_next_test": "restrict to PTGS2-high mesenchymal models; prostaglandin, myeloid polarization and immune-killing readouts",
            },
        ]
    )

    # Confirm that named repositioning partners really exist in the frozen
    # direction-compatible ChEMBL map for the intended state/target.
    named_checks = [
        ("hybrid-peak", "ROCK2", "BELUMOSUDIL"),
        ("hybrid-peak", "ROCK1", "FASUDIL"),
        ("mesenchymal-enriched", "PTGS2", "CELECOXIB"),
        ("mesenchymal-enriched", "FGFR1", "BRIVANIB"),
        ("mesenchymal-enriched", "KDR", "BRIVANIB"),
        ("mesenchymal-enriched", "FLT4", "BRIVANIB"),
    ]
    missing = [x for x in named_checks if not mechanism_available(mechanisms, *x)]
    if missing:
        raise ValueError(f"Named combination mechanisms missing from frozen map: {missing}")

    gates = pd.DataFrame(
        [
            {
                "state": "epithelial-enriched",
                "gate": "resolve_subtype_and_direction",
                "pass_condition": "same-direction raw and subtype-adjusted sensitivity in at least two screens or independent perturbation evidence",
                "current_status": "BLOCKED_FROM_COMBINATION_PROMOTION",
            },
            {
                "state": "hybrid-peak",
                "gate": "demonstrate_pair_synergy_and_immune_preservation",
                "pass_condition": "positive Bliss/ZIP in multiple hybrid-high models without loss of antigen presentation/T-cell viability",
                "current_status": "READY_FOR_COMPUTATIONAL_AND_EXPERIMENTAL_PAIR_TESTING",
            },
            {
                "state": "mesenchymal-enriched",
                "gate": "validate_remodelling_not_only_viability",
                "pass_condition": "reduced barrier/angiogenesis plus improved immune infiltration or killing in independent multicellular models",
                "current_status": "HYPOTHESIS_READY_NOT_BIOMARKER_VALIDATED",
            },
        ]
    )

    theme_counts = (
        pathways[pathways["representative"].astype(str).str.casefold().eq("true")]
        .groupby(["bulk_state_class", "biological_theme"])
        .size()
        .rename("representative_pathways")
        .reset_index()
    )
    target_context = targets[
        targets["gene"].isin({"ERBB2", "ERBB3", "ROCK1", "ROCK2", "FGFR1", "KDR", "FLT4", "PTGS2"})
    ].copy()

    anchors.to_csv(OUT / "integrated_state_anchor_evidence.tsv", sep="\t", index=False)
    combinations.to_csv(OUT / "state_specific_combination_hypotheses.tsv", sep="\t", index=False)
    gates.to_csv(OUT / "state_specific_decision_gates.tsv", sep="\t", index=False)
    theme_counts.to_csv(OUT / "representative_pathway_theme_counts.tsv", sep="\t", index=False)
    target_context.to_csv(OUT / "combination_target_pathway_context.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "locked evidence integration and state-specific combination hypothesis generation; not a synergy test",
        "states": {
            "epithelial-enriched": "unresolved; no combination promoted",
            "hybrid-peak": "ERBB2/ERBB3 dependency anchor validated; combination testing ready",
            "mesenchymal-enriched": "brivanib remodeling anchor retained as nominal convergent hypothesis",
        },
        "n_anchor_rows": int(len(anchors)),
        "n_combination_hypotheses": int(len(combinations)),
        "hybrid_erbb_class_support": h_class,
        "treatment_biomarker_established": False,
        "combination_synergy_demonstrated": False,
        "next_gate": "in-silico pair scoring and orthogonal target/drug literature anchoring before any therapeutic claim",
    }
    (OUT / "integrated_state_specific_combination_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
