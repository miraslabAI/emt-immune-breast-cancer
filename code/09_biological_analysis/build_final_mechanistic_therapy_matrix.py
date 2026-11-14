"""Integrate locked multiscale, topology, and Arm-C evidence into a therapy-hypothesis matrix."""
from __future__ import annotations

import os

import json
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LAND = ROOT / "04_analysis/08_biological_landscape"
SPATIAL = LAND / "spatial/NeoTRIP_outcome_free/paired_patient_spatial_state_contrasts.tsv"
TOPOLOGY = LAND / "spatial/NeoTRIP_EMT_topology/primary_topology_tests.tsv"
TOPOLOGY_SUMMARY = LAND / "spatial/NeoTRIP_EMT_topology/neotrip_emt_topology_summary.json"
MULTISCALE = LAND / "multiscale_synthesis/multiscale_synthesis_summary.json"
RESPONSE = ROOT / "04_analysis/09_treatment_context/NeoTRIP_Arm_C_spatial_response/primary_mechanism_pcr_associations.tsv"
RESPONSE_SUMMARY = ROOT / "04_analysis/09_treatment_context/NeoTRIP_Arm_C_spatial_response/neotrip_arm_c_spatial_response_summary.json"
OUT = ROOT / "04_analysis/10_mechanistic_model_and_therapy_hypotheses"


def fmt(x, digits=2):
    return f"{float(x):.{digits}f}"


def main():
    spatial = pd.read_csv(SPATIAL, sep="\t")
    topology = pd.read_csv(TOPOLOGY, sep="\t").set_index("estimand")
    multiscale = json.loads(MULTISCALE.read_text(encoding="utf-8"))
    topology_summary = json.loads(TOPOLOGY_SUMMARY.read_text(encoding="utf-8"))
    response = pd.read_csv(RESPONSE, sep="\t").set_index("score")
    response_summary = json.loads(RESPONSE_SUMMARY.read_text(encoding="utf-8"))
    if multiscale.get("status") != "PASS" or topology_summary.get("status") != "PASS":
        raise RuntimeError("Outcome-free biological prerequisites are incomplete")
    if topology_summary.get("pCR_or_Arm_read") is not False:
        raise RuntimeError("Topology analysis was not outcome-free")
    if response_summary.get("status") != "PASS" or response_summary.get("treatment_interaction_estimable") is not False:
        raise RuntimeError("Arm-C response evidence does not match the locked single-arm design")

    def spatial_hit(state, metric):
        contrast = f"{state}_minus_epithelial_dominant"
        hit = spatial.loc[(spatial.contrast == contrast) & (spatial.metric == metric)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing spatial contrast: {contrast}, {metric}")
        return hit.iloc[0]

    bridge = topology.loc["hybrid_bridge_contrast"]
    segregation = topology.loc["same_state_excess_fraction"]
    hvba = response.loc["HVBA"]
    mda = response.loc["MDA"]
    eca = response.loc["ECA"]

    rows = [
        {
            "priority": 1,
            "emt_ecosystem": "Hybrid E/M immune-visible barrier",
            "evidence_status": "Strong mechanism; suggestive chemotherapy-resistance direction; no treatment-benefit evidence",
            "convergent_evidence": "Bulk execution-low; malignant coactivation links antigen presentation/T-NK chemokines with TGF-beta/ECM; spatial HLA-ABC, PD-L1, T/Mono-Mac/stroma proximity and stromal barrier increased",
            "topology": f"Relative hybrid bridge supported (mean {fmt(bridge['mean'],3)}, FDR {fmt(bridge['fdr_bh'],4)}) within a segregated state landscape",
            "response_context": f"Arm-C HVBA OR for pCR {fmt(hvba['odds_ratio_per_sd'])}; permutation p {fmt(hvba['permutation_p'],3)}; FDR {fmt(hvba['fdr_bh'],3)} — hypothesis-generating only",
            "combination_hypothesis": "Checkpoint/immune-effector therapy plus TGF-beta–ECM or stromal-barrier modulation",
            "critical_validation": "Requires a dataset containing both chemotherapy and checkpoint-containing arms, or a prospective biomarker-stratified experiment",
        },
        {
            "priority": 2,
            "emt_ecosystem": "Mesenchymal immune-engaged, execution-impaired",
            "evidence_status": "Strong mechanism; chemotherapy-response association unresolved",
            "convergent_evidence": "Bulk suppressive recruitment/exclusion with reduced execution; malignant EMT/coactivation links TGF-beta/ECM and IFN response; spatial T, Mono/Mac and stroma proximity plus tumor HLA-ABC/PD-L1 increased while neighboring T cytotoxicity decreased",
            "topology": "Strong mesenchymal self-organization and depletion of direct epithelial–mesenchymal contacts",
            "response_context": f"Arm-C MDA OR for pCR {fmt(mda['odds_ratio_per_sd'])}; permutation p {fmt(mda['permutation_p'],3)}; FDR {fmt(mda['fdr_bh'],3)}",
            "combination_hypothesis": "Execution-restoring/checkpoint strategy plus microenvironment or stromal remodeling",
            "critical_validation": "Myeloid-targeting remains conditional because Mono/Mac suppressive-program intensity was not significant despite increased proximity",
        },
        {
            "priority": 3,
            "emt_ecosystem": "Epithelial-dominant cold/initiation-poor",
            "evidence_status": "Strong bulk ecosystem definition; spatial reference; chemotherapy-response association unresolved",
            "convergent_evidence": "Cold enriched and hot depleted across bulk cohorts; low antigen presentation/priming and effector recruitment",
            "topology": "Strong epithelial self-contact; direct epithelial–mesenchymal contact depleted",
            "response_context": f"Arm-C ECA OR for pCR {fmt(eca['odds_ratio_per_sd'])}; permutation p {fmt(eca['permutation_p'],3)}; FDR {fmt(eca['fdr_bh'],3)}",
            "combination_hypothesis": "Immune priming or immunogenic activation before/with checkpoint therapy",
            "critical_validation": "Checkpoint alone is not mechanistically prioritized in a poorly initiated cold ecosystem",
        },
        {
            "priority": 4,
            "emt_ecosystem": "Double-low indeterminate",
            "evidence_status": "Mixed and insufficient for therapy nomination",
            "convergent_evidence": "Bulk immune-rich associations conflict with lower spatial HLA-ABC and lower stromal barrier; purity/dilution and cell identity remain plausible",
            "topology": "Strong self-contact but marked depletion of hybrid contacts; biological identity remains unresolved",
            "response_context": "No EMT-state abundance association with Arm-C pCR",
            "combination_hypothesis": "No state-specific combination nominated",
            "critical_validation": "Resolve malignant identity, purity and technical dilution with orthogonal markers or transcriptomic mapping",
        },
    ]
    matrix = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT / "therapy_hypothesis_priority_matrix.tsv", sep="\t", index=False)

    report = f"""# Final multiscale EMT–immune mechanistic model v1.0

## Central model

Breast-tumour EMT is not coupled to a single cold-to-hot immune gradient. Instead, EMT states define distinct local immune bottlenecks. Mesenchymal and hybrid malignant states can be immune-visible and immune-contacted while failing to complete effective local cytotoxic execution.

## Spatial organization and the dynamic-EMT interpretation

- The malignant landscape is spatially segregated: same-state contacts exceed composition-based expectation by {fmt(100 * segregation['mean'],1)} percentage points (95% bootstrap CI {fmt(100 * segregation['bootstrap_ci95_low'],1)} to {fmt(100 * segregation['bootstrap_ci95_high'],1)}; FDR {fmt(segregation['fdr_bh'],4)}).
- Direct epithelial–mesenchymal contacts are depleted (mean log2 enrichment {fmt(topology.loc['direct_epithelial_mesenchymal_contact','mean'],3)}).
- Hybrid–mesenchymal contact is also mildly depleted, while epithelial–hybrid contact is approximately neutral.
- Nevertheless, the locked hybrid bridge contrast is positive ({fmt(bridge['mean'],3)}; 95% CI {fmt(bridge['bootstrap_ci95_low'],3)} to {fmt(bridge['bootstrap_ci95_high'],3)}; FDR {fmt(bridge['fdr_bh'],4)}). Hybrid E/M therefore acts as a **relative spatial bridge** compared with the more strongly separated direct epithelial–mesenchymal interface.
- This supports a compartmentalized EMT continuum but does not establish temporal transition or lineage direction.

## Immune coupling

Global EMT-state entropy and patient-level bridge strength showed no immune or barrier correlation surviving FDR. The supported biology is therefore state- and niche-specific, not a simple function of whole-tumour EMT diversity.

## Treatment-context result

The released spatial object contains only NeoTRIP Arm C (carboplatin plus nab-paclitaxel without atezolizumab), so checkpoint-treatment interaction is not identifiable. None of the three locked mechanism scores was significantly associated with Arm-C pCR after FDR. HVBA showed a directionally consistent but non-significant resistance signal (OR {fmt(hvba['odds_ratio_per_sd'])}; permutation p {fmt(hvba['permutation_p'],3)}; FDR {fmt(hvba['fdr_bh'],3)}), and must remain hypothesis-generating.

## Therapy-hypothesis hierarchy

1. **Hybrid immune-visible barrier:** strongest combination hypothesis — immune-effector/checkpoint therapy plus TGF-beta–ECM or stromal-barrier modulation.
2. **Mesenchymal immune-engaged, execution-impaired:** immune execution restoration plus microenvironment remodeling; myeloid targeting is conditional, not established.
3. **Epithelial cold/initiation-poor:** priming or immunogenic activation is biologically upstream of checkpoint efficacy.
4. **Double-low:** no therapy nomination until identity and purity are resolved.

These are mechanism-based hypotheses, not clinical treatment recommendations. Demonstrating treatment prediction requires an independent cohort with both control and relevant combination-treatment arms.
"""
    (OUT / "FINAL_MULTISCALE_EMT_IMMUNE_MECHANISTIC_MODEL_v1.0.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "PASS",
        "model": "state- and niche-specific immune engagement without assured immune completion",
        "spatial_dynamic_interpretation": "segregated EMT landscape with a relative hybrid bridge",
        "treatment_predictive_biomarker_established": False,
        "highest_priority_hypothesis": "Hybrid visible-barrier: immune-effector/checkpoint plus TGF-beta–ECM or stromal-barrier modulation",
        "next_gate": "target-and-drug literature anchoring followed by independent treatment-arm validation",
    }
    (OUT / "final_mechanistic_model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
