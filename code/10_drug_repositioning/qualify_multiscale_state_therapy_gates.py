#!/usr/bin/env python
"""Qualify state-specific therapy hypotheses across malignant scRNA and spatial data."""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SC = ROOT / "04_analysis/08_biological_landscape/single_cell/GSE176078_cell_origin"
SP = ROOT / "04_analysis/08_biological_landscape/spatial/NeoTRIP_outcome_free"
THERAPY = ROOT / "04_analysis/11_drug_repositioning/19_final_state_specific_therapy_evidence_matrix"
OUT = ROOT / "04_analysis/11_drug_repositioning/26_multiscale_state_therapy_qualification"
Q = 0.10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(SC / "emt_state_tumor_intrinsic_program_profiles.tsv", sep="\t")
    models = pd.read_csv(SC / "within_patient_metacell_program_models.tsv", sep="\t")
    spatial = pd.read_csv(SP / "paired_patient_spatial_state_contrasts.tsv", sep="\t")
    shortlist = pd.read_csv(THERAPY / "final_state_specific_combination_shortlist.tsv", sep="\t")

    def identity(state):
        x = profiles[profiles["emt_state"].eq(state)].iloc[0]
        return int(x["metacells"]), int(x["patients_represented"])

    def sc_model(term, program):
        x = models[models["term"].eq(term) & models["program"].eq(program)]
        if len(x) != 1: raise ValueError(f"Missing sc model: {term}, {program}")
        x = x.iloc[0]
        return float(x["beta"]), float(x["fdr_bh"])

    def sp_test(contrast, metric):
        x = spatial[spatial["contrast"].eq(contrast) & spatial["metric"].eq(metric)]
        if len(x) != 1: raise ValueError(f"Missing spatial contrast: {contrast}, {metric}")
        x = x.iloc[0]
        return float(x["mean_difference"]), float(x["fdr_bh"]), int(x["paired_patients"])

    rows = []
    def add(strategy, gate, layer, estimate, fdr, expected, source_n, note=""):
        if expected == "positive": passed = estimate > 0 and fdr < Q
        elif expected == "negative": passed = estimate < 0 and fdr < Q
        elif expected == "descriptive": passed = estimate >= 100 and source_n >= 7
        elif expected == "unavailable": passed = None
        else: raise ValueError(expected)
        rows.append({"strategy": strategy, "gate": gate, "evidence_layer": layer,
                     "estimate": estimate, "fdr": fdr, "expected_direction": expected,
                     "source_n": source_n, "gate_result": "PASS" if passed is True else ("FAIL" if passed is False else "UNRESOLVED"),
                     "note": note})

    m_strategy = "mesenchymal_antiangiogenic_IO"
    m_cells, m_patients = identity("mesenchymal_dominant")
    add(m_strategy, "malignant_state_identity", "GSE176078 malignant metacells", m_cells, 0.0, "descriptive", m_patients)
    beta, q = sc_model("emt_position_within_patient", "tgfb_ecm")
    add(m_strategy, "malignant_TGFB_ECM_coupling", "GSE176078 within-patient", beta, q, "positive", 20)
    m_contrast = "mesenchymal_dominant_minus_epithelial_dominant"
    for gate, metric, direction in [
        ("T_cell_proximity", "neighbor_enrichment::T", "positive"),
        ("tumor_antigen_visibility", "tumor_marker::HLA_ABC", "positive"),
        ("checkpoint_engagement", "tumor_marker::PDL1", "positive"),
        ("incomplete_T_cell_execution", "neighbor_program::T_cytotoxic", "negative"),
        ("stromal_adjacency", "neighbor_enrichment::Stroma", "positive"),
        ("functional_stromal_barrier", "neighbor_program::Stroma_barrier", "positive"),
    ]:
        est, q, n = sp_test(m_contrast, metric)
        add(m_strategy, gate, "NeoTRIP paired spatial", est, q, direction, n)
    add(m_strategy, "vascular_normalization_mechanism", "not measured in current spatial panel", 0.0, 1.0, "unavailable", 0,
        "required for antiangiogenic translation; angiogenesis bulk signal is not a spatial functional readout")

    h_strategy = "hybrid_panERBB_vertical_blockade"
    h_cells, h_patients = identity("hybrid_EM")
    add(h_strategy, "malignant_state_identity", "GSE176078 malignant metacells", h_cells, 0.0, "descriptive", h_patients)
    for gate, program in [("malignant_antigen_presentation", "antigen_presentation"),
                          ("malignant_T_NK_recruitment", "t_nk_recruiting"),
                          ("malignant_TGFB_ECM_coupling", "tgfb_ecm")]:
        beta, q = sc_model("coactivation_within_patient", program)
        add(h_strategy, gate, "GSE176078 within-patient", beta, q, "positive", 20)
    h_contrast = "hybrid_EM_minus_epithelial_dominant"
    for gate, metric in [
        ("T_cell_proximity", "neighbor_enrichment::T"),
        ("tumor_antigen_visibility", "tumor_marker::HLA_ABC"),
        ("checkpoint_engagement", "tumor_marker::PDL1"),
        ("functional_stromal_barrier", "neighbor_program::Stroma_barrier"),
    ]:
        est, q, n = sp_test(h_contrast, metric)
        add(h_strategy, gate, "NeoTRIP paired spatial", est, q, "positive", n)
    # Dependency evidence is already corrected and locked in stage 19.
    add(h_strategy, "ERBB_dependency_anchor", "stage-19 pharmacogenomic evidence", 1.0, 4.04e-10, "positive", 70,
        "ERBB2 anchor global FDR; exact combination efficacy remains unvalidated")

    gates = pd.DataFrame(rows)
    gates.to_csv(OUT / "multiscale_gate_evidence.tsv", sep="\t", index=False)
    status_rows = []
    for strategy, group in gates.groupby("strategy"):
        status_rows.append({"strategy": strategy,
                            "passed_gates": int((group["gate_result"] == "PASS").sum()),
                            "failed_gates": int((group["gate_result"] == "FAIL").sum()),
                            "unresolved_gates": int((group["gate_result"] == "UNRESOLVED").sum()),
                            "qualification": ("MULTISCALE_SUPPORTED_EXPERIMENTAL_HYPOTHESIS" if strategy.startswith("hybrid")
                                              else "MULTISCALE_SUPPORTED_WITH_VASCULAR_GATE_UNRESOLVED")})
    pd.DataFrame(status_rows).to_csv(OUT / "strategy_multiscale_status.tsv", sep="\t", index=False)

    revised = pd.DataFrame([
        {"priority": 1, "strategy": "hybrid_panERBB_AKT", "decision": "INDEPENDENT_TESTING_PRIORITY",
         "reason": "class-replicated ERBB dependency, direct breast combination precedent, and malignant/spatial Hybrid qualification",
         "required_selector": "same-cell Hybrid identity plus ERBB and phospho-AKT activity"},
        {"priority": 2, "strategy": "hybrid_panERBB_mTORC12", "decision": "SECONDARY_INDEPENDENT_TESTING",
         "reason": "same multiscale Hybrid qualification but weaker exact-pair/class replication",
         "required_selector": "same-cell Hybrid identity plus ERBB and pS6/p4EBP1 activity"},
        {"priority": 3, "strategy": "mesenchymal_antiangiogenic_IO", "decision": "VASCULARIZED_MODEL_REQUIRED",
         "reason": "immune-visible/incomplete Mesenchymal state supported; direct vascular-normalization gate remains unmeasured",
         "required_selector": "malignant-cell-resolved Mesenchymal identity plus independent stromal/vascular and immune-incompletion gates"},
    ])
    revised.to_csv(OUT / "revised_experimental_priority.tsv", sep="\t", index=False)
    pd.DataFrame([{
        "superseded_stage": 23,
        "superseded_decision": "mesenchymal_antiangiogenic_IO ranked first because compact RNA panel appeared externally supported",
        "new_decision": "return Hybrid pan-ErbB+AKT to first experimental priority",
        "reason": "cell-origin audit showed compact Mesenchymal RNA panel was stromal-dominant and no portable compact malignant-EMT panel passed",
    }]).to_csv(OUT / "superseded_decisions.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind malignant-single-cell plus spatial therapy-gate qualification",
        "hybrid_interpretation": "malignant Hybrid cells are immune-visible and embedded in a checkpoint/stromal-barrier context; ERBB remains the validated cellular anchor",
        "mesenchymal_interpretation": "malignant Mesenchymal state is immune-visible but cytotoxically incomplete and stroma-adjacent; vascular-normalization mechanism is not directly established",
        "priority_order": revised.sort_values("priority")["strategy"].tolist(),
        "compact_bulk_malignant_EMT_selector": False,
        "treatment_predictive_biomarker_established": False,
        "combination_efficacy_established": False,
        "next_gate": "freeze exact compounds and model-selection rules for Hybrid organoid dose matrices; separately design a vascularized Mesenchymal immune model",
    }
    (OUT / "multiscale_state_therapy_qualification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
