#!/usr/bin/env python
"""Build the final, claim-calibrated EMT-state therapy evidence matrix.

The output distinguishes internal discovery, same-dataset robustness, external
mechanistic literature, clinical precedent, negative evidence, and what remains
untested.  No numerical score is used to hide disagreement between evidence
layers.
"""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
ROBUST = BASE / "18_hybrid_drugcomb_robustness/hybrid_pair_subtype_loo_robustness.tsv"
CLASS_ROBUST = BASE / "18_hybrid_drugcomb_robustness/hybrid_partner_class_level_robustness.tsv"
DRUGCOMB = BASE / "17_state_specific_drugcomb_synergy/state_specific_drugcomb_synergy_synthesis.tsv"
TARGET_FIRST = BASE / "13_target_first_dependency_screen/target_first_state_dependency_synthesis.tsv"
PGX = BASE / "11_pharmacogenomic_validation/state_specific_pharmacogenomic_synthesis.tsv"
OUT = BASE / "19_final_state_specific_therapy_evidence_matrix"


def get_row(frame: pd.DataFrame, **conditions: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).str.casefold().eq(value.casefold())
    hit = frame[mask]
    if hit.empty:
        raise ValueError(f"Required evidence row missing: {conditions}")
    return hit.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    robust = pd.read_csv(ROBUST, sep="\t", low_memory=False)
    class_robust = pd.read_csv(CLASS_ROBUST, sep="\t", low_memory=False)
    drugcomb = pd.read_csv(DRUGCOMB, sep="\t", low_memory=False)
    target_first = pd.read_csv(TARGET_FIRST, sep="\t", low_memory=False)
    pgx = pd.read_csv(PGX, sep="\t", low_memory=False)

    azd2014 = get_row(robust, anchor="Sapitinib", partner="AZD2014")
    mk2206 = get_row(robust, anchor="Sapitinib", partner="MK-2206")
    erbb_anchor = get_row(target_first, state="hybrid-peak", drug="BMS-690514")
    brivanib = get_row(pgx, state="mesenchymal-enriched", candidate="brivanib")
    celecoxib = drugcomb[
        drugcomb["analysis_family"].eq("mesenchymal_antiangiogenic_partner_discovery")
        & drugcomb["prespecified_partner"].astype(str).str.casefold().eq("true")
    ]
    if len(celecoxib) != 3:
        raise ValueError("Expected three frozen antiangiogenic-celecoxib pairs")

    literature = pd.DataFrame(
        [
            {
                "evidence_id": "LIT_H_AKT_01",
                "axis": "pan-ErbB + AKT inhibition",
                "model_or_population": "HER2-amplified breast cancer cells and HCC1954 xenograft",
                "finding": "AZD8931 plus capivasertib/AZD5363 produced synergistic growth inhibition and deeper regressions than either agent alone",
                "evidence_level": "direct_preclinical_breast_combination",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "26095475",
                "doi": "10.3892/ijo.2015.3062",
                "url": "https://pubmed.ncbi.nlm.nih.gov/26095475/",
            },
            {
                "evidence_id": "LIT_H_AKT_02",
                "axis": "HER2 inhibition + MK-2206",
                "model_or_population": "advanced HER2-positive breast cancer/solid tumors",
                "finding": "lapatinib plus MK-2206 was clinically feasible in phase I, with toxicity constraints and limited expansion efficacy data",
                "evidence_level": "early_phase_clinical_precedent",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "27026198",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27026198/",
            },
            {
                "evidence_id": "LIT_H_AKT_03",
                "axis": "trastuzumab/paclitaxel + MK-2206",
                "model_or_population": "advanced HER2-amplified malignancies, predominantly breast",
                "finding": "phase Ib established a tolerable schedule and reported preliminary activity despite prior HER2 therapy",
                "evidence_level": "early_phase_clinical_precedent",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "26875185",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/26875185/",
            },
            {
                "evidence_id": "LIT_H_MTOR_01",
                "axis": "dual mTORC1/2 inhibition",
                "model_or_population": "ER-positive breast cancer models",
                "finding": "AZD2014/vistusertib showed broad antiproliferative activity and combination activity with fulvestrant; not evidence for the exact pan-ErbB pair",
                "evidence_level": "drug_and_pathway_preclinical_support",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "26358751",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/26358751/",
            },
            {
                "evidence_id": "LIT_M_IO_01",
                "axis": "VEGFR2 inhibition + PD-1 blockade",
                "model_or_population": "syngeneic breast models and advanced TNBC",
                "finding": "low-dose VEGFR2 blockade improved immune infiltration and sensitized tumors to PD-1 blockade; effect was dose-dependent",
                "evidence_level": "direct_preclinical_plus_early_clinical_breast_combination",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "31848190",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/31848190/",
            },
            {
                "evidence_id": "LIT_M_IO_02",
                "axis": "angiogenesis inhibition + PD-1 + chemotherapy",
                "model_or_population": "immunomodulatory advanced TNBC, phase II FUTURE-C-Plus",
                "finding": "famitinib, camrelizumab and chemotherapy showed clinical activity; CD8/PD-L1 positive tumors benefited more",
                "evidence_level": "phase_II_clinical_class_precedent",
                "supports_state_specificity": False,
                "negative_or_cautionary": False,
                "pmid": "35337339",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/35337339/",
            },
            {
                "evidence_id": "LIT_H_NEG_01",
                "axis": "AZD8931 + endocrine therapy",
                "model_or_population": "unselected endocrine-therapy-naive advanced HR-positive breast cancer, randomized phase II",
                "finding": "no PFS or secondary-endpoint benefit and greater skin/gastrointestinal toxicity",
                "evidence_level": "randomized_phase_II_negative_context",
                "supports_state_specificity": False,
                "negative_or_cautionary": True,
                "pmid": "27654971",
                "doi": "10.1007/s10549-016-3979-5",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27654971/",
            },
        ]
    )
    literature.to_csv(OUT / "orthogonal_literature_evidence.tsv", sep="\t", index=False)

    shortlist = pd.DataFrame(
        [
            {
                "rank": 1,
                "state": "hybrid-peak",
                "therapy_hypothesis": "pan-ErbB blockade + AKT inhibition",
                "discovery_leads": "sapitinib + MK-2206; translational analogues should be selected by current development status",
                "internal_anchor_evidence": f"ERBB anchor global FDR={float(erbb_anchor['best_global_fdr']):.3g}; class-level dependency replication",
                "internal_pair_evidence": f"DrugComb nominal p={float(mk2206['discovery_consensus_p']):.3g}, global FDR={float(mk2206['discovery_global_fdr']):.3g}; subtype-adjusted p={float(mk2206['subtype_adjusted_one_sided_p']):.3g}",
                "external_evidence": "direct AZD8931+AKT-inhibitor preclinical breast synergy plus phase-I HER2+ precedents",
                "negative_evidence": "same-dataset class-level FDR negative; sapitinib failed in unselected HR+/HER2-low phase-II contexts",
                "evidence_tier": "Tier_B_biomarker_defined_translational_hypothesis",
                "claim_allowed": "Hybrid-state prioritized combination hypothesis",
                "claim_not_allowed": "validated predictive biomarker or clinically effective regimen",
                "biomarker_gate": "hybrid EMT coactivation plus active ERBB2/3 and PI3K-AKT signaling; do not use HER2 status alone",
                "next_validation": "independent hybrid-high organoid/PDX panel with dose matrices, pAKT/pS6/p4EBP1 and immune-preservation endpoints",
            },
            {
                "rank": 2,
                "state": "hybrid-peak",
                "therapy_hypothesis": "pan-ErbB blockade + dual mTORC1/2 inhibition",
                "discovery_leads": "sapitinib + AZD2014/vistusertib",
                "internal_anchor_evidence": "validated ERBB-family state dependency anchor",
                "internal_pair_evidence": f"DrugComb nominal p={float(azd2014['discovery_consensus_p']):.3g}, global FDR={float(azd2014['discovery_global_fdr']):.3g}; subtype-adjusted p={float(azd2014['subtype_adjusted_one_sided_p']):.3g}; basal-A delta={float(azd2014['basal_A_cliffs_delta']):.2f}",
                "external_evidence": "mTORC1/2 drug activity in breast models and general HER/PI3K-mTOR vertical-blockade rationale; exact pair not independently validated",
                "negative_evidence": "no class-level FDR and afatinib+AZD2014 did not reproduce the adjusted direction",
                "evidence_tier": "Tier_B_minus_anchor_specific_discovery_hypothesis",
                "claim_allowed": "second-priority Hybrid experimental pair",
                "claim_not_allowed": "class-wide synergy",
                "biomarker_gate": "hybrid EMT plus ERBB activation plus pS6/p4EBP1 or mTORC1/2 activity",
                "next_validation": "head-to-head pan-ErbB anchors with AZD2014; quantify whether effect is sapitinib-specific",
            },
            {
                "rank": 3,
                "state": "mesenchymal-enriched",
                "therapy_hypothesis": "low-dose FGFR/VEGFR-axis remodeling + PD-1/PD-L1 blockade",
                "discovery_leads": "brivanib as computational discovery anchor; use a clinically viable anti-angiogenic analogue for translation",
                "internal_anchor_evidence": f"brivanib one-screen nominal viability and LINCS/mechanism convergence; effect={float(brivanib['lincs_evidence_adjusted_score']):.3g}",
                "internal_pair_evidence": "not evaluable in tumor-cell-only DrugComb",
                "external_evidence": "breast/TNBC preclinical and phase-II class precedent for anti-VEGF/VEGFR plus PD-1",
                "negative_evidence": "Mesenchymal specificity has not been clinically shown; dose and vascular normalization window are critical",
                "evidence_tier": "Tier_B_mechanistic_ecosystem_hypothesis",
                "claim_allowed": "Mesenchymal/stromal-barrier remodeling hypothesis",
                "claim_not_allowed": "brivanib-specific synergy or established EMT treatment biomarker",
                "biomarker_gate": "mesenchymal EMT plus angiogenic/ECM barrier plus immune engagement without completion",
                "next_validation": "multicellular vascularized tumor-immune model; infiltration, perfusion, IFNG/GZMB and killing rather than viability alone",
            },
            {
                "rank": 4,
                "state": "hybrid-peak",
                "therapy_hypothesis": "pan-ErbB blockade + PD-1/PD-L1 blockade",
                "discovery_leads": "ERBB inhibitor selected by biomarker and immune-compatible exposure",
                "internal_anchor_evidence": "strong ERBB dependency plus Hybrid immune visibility/incomplete execution",
                "internal_pair_evidence": "not evaluable in tumor-cell-only DrugComb",
                "external_evidence": "component-level rationale only in the current evidence lock",
                "negative_evidence": "pan-ErbB toxicity and possible immune-cell effects require direct testing",
                "evidence_tier": "Tier_C_mechanistic_pair",
                "claim_allowed": "immune-completion hypothesis",
                "claim_not_allowed": "synergy claim",
                "biomarker_gate": "hybrid EMT, ERBB activity, antigen presentation and pre-existing T-cell engagement",
                "next_validation": "tumor-immune co-culture with antigen-presentation and T-cell fitness readouts",
            },
        ]
    )
    shortlist.to_csv(OUT / "final_state_specific_combination_shortlist.tsv", sep="\t", index=False)

    state_matrix = pd.DataFrame(
        [
            {
                "state": "epithelial-enriched",
                "dominant_biology": "oxidative/peroxisomal metabolism, protein homeostasis and growth signaling",
                "validated_anchor": "none",
                "combination_status": "NO_COMBINATION_PROMOTED",
                "reason": "single-agent hits were screen-specific or directionally inconsistent after subtype adjustment",
                "priority_experiment": "resolve metabolic dependency versus adaptation in lineage-matched models",
            },
            {
                "state": "hybrid-peak",
                "dominant_biology": "ERBB2/3-PTK6 signaling with RHO/adhesion/motility and immune visibility",
                "validated_anchor": "ERBB2/ERBB3 inhibitor class",
                "combination_status": "PRIORITIZED_FOR_INDEPENDENT_TESTING",
                "reason": "class-replicated single-agent dependency plus two directionally robust but non-FDR DrugComb pairs",
                "priority_experiment": "pan-ErbB plus AKT/mTOR dose matrices in hybrid-high models",
            },
            {
                "state": "mesenchymal-enriched",
                "dominant_biology": "ECM/integrin, hypoxia/angiogenesis, inflammatory signaling and immune incompletion",
                "validated_anchor": "none; brivanib is a nominal remodeling anchor",
                "combination_status": "ECOSYSTEM_HYPOTHESIS",
                "reason": "cell-autonomous synergy absent; external evidence supports anti-angiogenic immune remodeling at class level",
                "priority_experiment": "vascular-stromal-immune validation of low-dose anti-angiogenic plus checkpoint blockade",
            },
        ]
    )
    state_matrix.to_csv(OUT / "final_state_therapy_decision_matrix.tsv", sep="\t", index=False)

    downranked = pd.DataFrame(
        [
            {
                "hypothesis": "anti-angiogenic + celecoxib in Mesenchymal",
                "decision": "DOWNRANK",
                "reason": "three prespecified ALMANAC pairs had no corrected support; axitinib and sunitinib showed predominantly opposite state direction",
            },
            {
                "hypothesis": "pan-ErbB + ROCK inhibitor",
                "decision": "UNTESTED_TIER_C",
                "reason": "biological complementarity remains, but no direct DrugComb pair was observed",
            },
            {
                "hypothesis": "universal EMT-state regimen",
                "decision": "REJECT",
                "reason": "state pathways, compartments and drug evidence are non-interchangeable",
            },
            {
                "hypothesis": "Epithelial AR/triapine combination",
                "decision": "REJECT_FOR_NOW",
                "reason": "single-screen/covariate-dependent results with raw-direction inconsistency",
            },
        ]
    )
    downranked.to_csv(OUT / "downranked_or_rejected_hypotheses.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "final_model": "EMT-state-specific malignant dependency plus ecosystem immune-completion framework",
        "top_hybrid_hypothesis": "pan-ErbB plus AKT inhibition in biomarker-defined hybrid tumors",
        "second_hybrid_hypothesis": "pan-ErbB plus dual mTORC1/2 inhibition",
        "top_mesenchymal_hypothesis": "low-dose FGFR/VEGFR-axis remodeling plus PD-1/PD-L1 blockade",
        "epithelial_status": "unresolved; no combination promoted",
        "celecoxib_status": "downranked",
        "validated_combination_synergy": False,
        "validated_treatment_predictive_biomarker": False,
        "next_gate": "prospective or independent state-stratified combination validation with immune-aware endpoints",
    }
    (OUT / "final_state_specific_therapy_evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
