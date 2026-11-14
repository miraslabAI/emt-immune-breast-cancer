#!/usr/bin/env python
"""Build the final outcome-blind EMT-state therapy atlas and claim-boundary package."""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
OUT = BASE / "31_final_computational_evidence_synthesis"


def load_json(relative: str) -> dict:
    return json.loads((BASE / relative).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    pathway = load_json("06_collapsed_state_pathway_map/collapsed_state_pathway_map_summary.json")
    pharmacogenomics = load_json("11_pharmacogenomic_validation/pharmacogenomic_validation_summary.json")
    dependency = load_json("13_target_first_dependency_screen/target_first_dependency_screen_summary.json")
    combination = load_json("17_state_specific_drugcomb_synergy/state_specific_drugcomb_synergy_summary.json")
    robustness = load_json("18_hybrid_drugcomb_robustness/hybrid_drugcomb_robustness_summary.json")
    final_matrix = load_json("19_final_state_specific_therapy_evidence_matrix/final_state_specific_therapy_evidence_summary.json")
    scores = load_json("21_state_specific_therapy_scores/therapy_score_model_summary.json")
    multiscale = load_json("26_multiscale_state_therapy_qualification/multiscale_state_therapy_qualification_summary.json")
    compounds = load_json("27_hybrid_compound_model_lock/hybrid_compound_model_lock_summary.json")
    plate_lock = load_json("28_hybrid_plate_analysis_lock/hybrid_plate_analysis_lock_summary.json")
    ingestion = load_json("29_hybrid_result_ingestion_contract/result_ingestion_contract_summary.json")

    for name, summary in {
        "pathway": pathway, "pharmacogenomics": pharmacogenomics, "dependency": dependency,
        "combination": combination, "robustness": robustness, "final_matrix": final_matrix,
        "scores": scores, "multiscale": multiscale, "compounds": compounds,
        "plate_lock": plate_lock, "ingestion": ingestion,
    }.items():
        require(summary.get("status") in {"PASS", "REVIEW"}, f"Upstream stage {name} is not finalized")

    require(combination["primary_global_fdr_lt_0.10"] == 0,
            "The final atlas assumes no multiplicity-corrected DrugComb pair")
    require(not final_matrix["validated_combination_synergy"],
            "Claim boundary changed: combination synergy is now validated")
    require(not final_matrix["validated_treatment_predictive_biomarker"],
            "Claim boundary changed: a predictive biomarker is now validated")
    require(not multiscale["combination_efficacy_established"],
            "Claim boundary changed: combination efficacy is now established")
    require(scores["all_rank_stability_gates_passed"],
            "Therapy-hypothesis scores did not pass the prespecified stability gate")
    require(ingestion["response_data_used"] is False,
            "The prospective result-ingestion contract must remain outcome blind")

    atlas = pd.DataFrame([
        {
            "priority": 1,
            "emt_state": "hybrid-peak",
            "strategy_id": "hybrid_panERBB_AKT",
            "strategy": "pan-ErbB blockade plus AKT inhibition",
            "reference_pair": "sapitinib + capivasertib",
            "translational_pair": "neratinib + capivasertib",
            "state_biology": "malignant Hybrid identity; ERBB2/3 dependency; immune visibility with checkpoint and stromal-barrier context",
            "bulk_pathway_support": "supported",
            "malignant_single_cell_support": "supported",
            "spatial_ecosystem_support": "supported",
            "target_dependency_support": "global-FDR ERBB anchor with class replication",
            "perturbation_support": "mechanism-consistent LINCS and pharmacogenomic support",
            "combination_dataset_support": "directionally robust; not multiplicity-corrected or independent",
            "exact_pair_external_anchor": "direct breast preclinical sapitinib-capivasertib precedent",
            "evidence_grade": "B1",
            "decision": "PRIMARY_INDEPENDENT_TESTING_PRIORITY",
            "required_selector": "same-cell Hybrid identity plus ERBB and phospho-AKT activity",
            "allowed_claim": "biomarker-defined experimental combination hypothesis",
            "prohibited_claim": "validated predictive biomarker, clinical efficacy, or treatment recommendation",
            "next_validation": "locked 36-plate prospective dose-matrix study with phosphoprotein and apoptosis gates",
        },
        {
            "priority": 2,
            "emt_state": "hybrid-peak",
            "strategy_id": "hybrid_panERBB_mTORC12",
            "strategy": "pan-ErbB blockade plus dual mTORC1/2 inhibition",
            "reference_pair": "sapitinib + vistusertib",
            "translational_pair": "neratinib + vistusertib",
            "state_biology": "same multiscale Hybrid qualification with ERBB-to-mTOR vertical signaling rationale",
            "bulk_pathway_support": "supported",
            "malignant_single_cell_support": "supported",
            "spatial_ecosystem_support": "supported",
            "target_dependency_support": "validated ERBB anchor; mTOR partner less state-selective",
            "perturbation_support": "mechanistically coherent",
            "combination_dataset_support": "nominal discovery signal; no corrected or class-wide replication",
            "exact_pair_external_anchor": "general HER/PI3K-mTOR rationale; exact pair not independently validated",
            "evidence_grade": "B2",
            "decision": "SECONDARY_INDEPENDENT_TESTING",
            "required_selector": "same-cell Hybrid identity plus ERBB and pS6/p4EBP1 activity",
            "allowed_claim": "second-priority anchor-specific experimental hypothesis",
            "prohibited_claim": "class-wide synergy or clinical efficacy",
            "next_validation": "run only after the AKT primary gate; compare sapitinib and neratinib anchors",
        },
        {
            "priority": 3,
            "emt_state": "mesenchymal-enriched",
            "strategy_id": "mesenchymal_antiangiogenic_IO",
            "strategy": "low-dose FGFR/VEGFR-axis remodeling plus PD-1/PD-L1 blockade",
            "reference_pair": "FGFR/VEGFR remodeling agent + checkpoint blockade",
            "translational_pair": "select clinically viable agents after vascular-model qualification",
            "state_biology": "malignant Mesenchymal identity; ECM/angiogenic barrier; immune visibility with incomplete cytotoxic execution",
            "bulk_pathway_support": "supported",
            "malignant_single_cell_support": "supported",
            "spatial_ecosystem_support": "partly supported; vascular-normalization mechanism unmeasured",
            "target_dependency_support": "no validated cell-autonomous anchor",
            "perturbation_support": "brivanib convergence is nominal and remodeling-oriented",
            "combination_dataset_support": "not evaluable in tumor-cell-only DrugComb",
            "exact_pair_external_anchor": "anti-angiogenic plus checkpoint class precedent; state specificity unvalidated",
            "evidence_grade": "B2E",
            "decision": "VASCULARIZED_MODEL_REQUIRED",
            "required_selector": "malignant Mesenchymal identity plus independent vascular/stromal and immune-incompletion gates",
            "allowed_claim": "mechanistic ecosystem-remodeling hypothesis",
            "prohibited_claim": "brivanib-specific synergy, vascular normalization, or established EMT treatment biomarker",
            "next_validation": "vascularized multicellular tumor-immune model with perfusion, infiltration, IFNG/GZMB and killing readouts",
        },
        {
            "priority": 4,
            "emt_state": "hybrid-peak",
            "strategy_id": "hybrid_panERBB_IO",
            "strategy": "pan-ErbB blockade plus PD-1/PD-L1 blockade",
            "reference_pair": "not frozen",
            "translational_pair": "select only after immune-compatibility testing",
            "state_biology": "Hybrid ERBB dependency plus antigen visibility and incomplete immune execution",
            "bulk_pathway_support": "supported",
            "malignant_single_cell_support": "supported",
            "spatial_ecosystem_support": "supported",
            "target_dependency_support": "ERBB anchor supported",
            "perturbation_support": "component-level rationale only",
            "combination_dataset_support": "not evaluable in tumor-cell-only data",
            "exact_pair_external_anchor": "not established in the frozen project evidence",
            "evidence_grade": "C",
            "decision": "EXPLORATORY_IMMUNE_COMPLETION_HYPOTHESIS",
            "required_selector": "Hybrid state, active ERBB, antigen presentation and pre-existing T-cell engagement",
            "allowed_claim": "immune-aware exploratory hypothesis",
            "prohibited_claim": "synergy or clinical recommendation",
            "next_validation": "tumor-immune co-culture with antigen presentation and T-cell fitness endpoints",
        },
        {
            "priority": pd.NA,
            "emt_state": "epithelial-enriched",
            "strategy_id": "epithelial_unresolved",
            "strategy": "no combination promoted",
            "reference_pair": "none",
            "translational_pair": "none",
            "state_biology": "oxidative/peroxisomal metabolism, protein homeostasis and growth signaling",
            "bulk_pathway_support": "supported",
            "malignant_single_cell_support": "insufficient for a therapy anchor",
            "spatial_ecosystem_support": "not therapy-qualifying",
            "target_dependency_support": "screen-specific or covariate-dependent signals",
            "perturbation_support": "insufficiently convergent",
            "combination_dataset_support": "no robust state-specific combination",
            "exact_pair_external_anchor": "none",
            "evidence_grade": "U",
            "decision": "UNRESOLVED_NO_COMBINATION_PROMOTED",
            "required_selector": "not defined",
            "allowed_claim": "state biology is distinct and requires separate dependency resolution",
            "prohibited_claim": "forced therapy assignment or universal EMT regimen",
            "next_validation": "resolve metabolic dependency versus adaptation in lineage-matched models",
        },
    ])
    atlas.to_csv(OUT / "final_state_specific_therapy_atlas.tsv", sep="\t", index=False)

    grades = pd.DataFrame([
        ["A", "independently replicated state-stratified combination efficacy", "none in the current study"],
        ["B1", "multiscale state qualification, validated dependency anchor, exact-pair external precedent and prospective test lock", "Hybrid pan-ErbB + AKT"],
        ["B2", "multiscale qualification and coherent anchor, but exact-pair or partner replication remains incomplete", "Hybrid pan-ErbB + mTORC1/2"],
        ["B2E", "multiscale ecosystem hypothesis with a required functional gate still unresolved", "Mesenchymal antiangiogenic + checkpoint"],
        ["C", "component-level mechanistic hypothesis without direct pair validation", "Hybrid pan-ErbB + checkpoint"],
        ["U", "state biology resolved but no therapy hypothesis meets promotion criteria", "Epithelial state"],
    ], columns=["grade", "definition", "assigned_strategy"])
    grades.to_csv(OUT / "evidence_grade_legend.tsv", sep="\t", index=False)

    boundaries = pd.DataFrame([
        ["EMT states have distinct pathway and immune-ecosystem configurations", True, "replicated bulk, single-cell and spatial evidence"],
        ["Hybrid tumors are prioritized for pan-ErbB vertical-blockade experiments", True, "validated ERBB anchor plus multiscale Hybrid qualification"],
        ["Mesenchymal tumors motivate vascular/immune remodeling experiments", True, "ecosystem evidence; vascular mechanism remains unresolved"],
        ["A treatment-predictive biomarker has been established", False, "no independent treatment-arm validation"],
        ["Any proposed combination has validated efficacy", False, "no prospective response data and no corrected independent combination validation"],
        ["One regimen should be used across all EMT states", False, "state biology and evidence are non-interchangeable"],
        ["The compact bulk panel is a malignant-cell selector", False, "cellular-origin and portability gates failed"],
        ["The therapy scores rank hypothesis eligibility", True, "outcome-blind and stable across tested weights"],
        ["The therapy scores predict patient response", False, "weights were not fit to treatment outcomes"],
    ], columns=["statement", "allowed", "basis"])
    boundaries.to_csv(OUT / "final_claim_boundary_matrix.tsv", sep="\t", index=False)

    states = pd.DataFrame([
        ["epithelial-enriched", pathway["representatives_by_state"]["epithelial-enriched"], "metabolic adaptation, protein/RNA homeostasis and growth signaling", "unresolved", "no combination promoted"],
        ["hybrid-peak", pathway["representatives_by_state"]["hybrid-peak"], "adhesion/ECM motility plus ERBB growth-survival signaling", "malignant and spatially qualified", "pan-ErbB + AKT first; pan-ErbB + mTORC1/2 second"],
        ["mesenchymal-enriched", pathway["representatives_by_state"]["mesenchymal-enriched"], "ECM/integrin, inflammatory, hypoxia/angiogenesis and immune incompletion", "malignant identity supported; vascular function unresolved", "antiangiogenic remodeling + checkpoint in vascularized models"],
    ], columns=["emt_state", "nonredundant_pathway_representatives", "dominant_biology", "multiscale_status", "therapy_position"])
    states.to_csv(OUT / "final_state_biology_summary.tsv", sep="\t", index=False)

    negative = pd.DataFrame([
        ["global state-selective pharmacogenomic validation", pharmacogenomics["global_fdr_lt_0.10"], "no candidate passed global FDR < 0.10"],
        ["DrugComb state-specific synergy", combination["primary_global_fdr_lt_0.10"], "no locked pair passed global FDR < 0.10"],
        ["Hybrid subtype-adjusted robustness", robustness["subtype_adjusted_fdr_lt_0.10"], "directional robustness is not corrected validation"],
        ["compact bulk malignant EMT selector", 0, "rejected because of cellular-origin and cross-cohort portability limits"],
        ["Epithelial combination", 0, "no combination met promotion criteria"],
    ], columns=["negative_or_null_result", "count_passing_primary_gate", "interpretation"])
    negative.to_csv(OUT / "important_negative_evidence.tsv", sep="\t", index=False)

    report = f"""# Final computational evidence synthesis

## Central model

Breast-cancer EMT is represented as a state-specific malignant-dependency and immune-ecosystem continuum. The analysis does not support a universal EMT regimen.

## Ranked therapeutic hypotheses

1. Hybrid pan-ErbB plus AKT inhibition is the primary independent-testing priority. The reference pair is {compounds['primary_reference_pair']}; the translational class analogue is {compounds['primary_translational_analogue']}.
2. Hybrid pan-ErbB plus dual mTORC1/2 inhibition is a secondary experimental hypothesis with weaker exact-pair replication.
3. Mesenchymal antiangiogenic plus checkpoint therapy is an ecosystem-remodeling hypothesis that requires a vascularized immune model; vascular normalization is not established by the current data.
4. No Epithelial combination is promoted.

## What the study establishes

- EMT states have distinct, replicated biological and immune contexts.
- Hybrid-state ERBB dependence is the strongest malignant-cell therapy anchor.
- Single-cell and spatial analyses qualify the Hybrid and Mesenchymal contexts for different experimental strategies.
- A complete prospective 36-plate result-ingestion and synergy-analysis contract has been frozen without using response outcomes.

## What the study does not establish

- No treatment-predictive biomarker is validated.
- No proposed combination has established efficacy.
- No therapy score should be interpreted as a patient-response predictor.
- No universal treatment should be assigned across EMT states.
"""
    (OUT / "FINAL_COMPUTATIONAL_EVIDENCE_SYNTHESIS.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "final outcome-blind EMT-state-specific therapy evidence synthesis",
        "central_model": "state-specific malignant dependency plus immune-ecosystem completion",
        "state_decisions": {
            "epithelial-enriched": "unresolved; no combination promoted",
            "hybrid-peak": "pan-ErbB + AKT primary; pan-ErbB + dual mTORC1/2 secondary",
            "mesenchymal-enriched": "antiangiogenic remodeling + checkpoint requires vascularized immune validation",
        },
        "top_reference_pair": compounds["primary_reference_pair"],
        "top_translational_pair": compounds["primary_translational_analogue"],
        "prospective_plates_locked": plate_lock["primary_screen_plates"],
        "response_data_used_for_final_synthesis": False,
        "validated_treatment_predictive_biomarker": False,
        "validated_combination_efficacy": False,
        "clinical_recommendation": False,
        "next_gate": "generate paper-ready figures and manuscript result modules from the frozen atlas",
    }
    (OUT / "final_computational_evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
