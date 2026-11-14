#!/usr/bin/env python
"""Freeze the tiered assay and prospective validation blueprint; performs no experiments."""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis/11_drug_repositioning/23_prospective_validation_blueprint"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    strategy = [
        {
            "strategy": "mesenchymal_antiangiogenic_IO", "priority": 1,
            "status": "compact RNA panel externally supported",
            "selection": "high mesenchymal + vascular/stromal barrier + immune visibility + incomplete completion",
            "intervention": "low-dose FGFR/VEGFR remodeling plus PD-1/PD-L1 blockade",
            "required_model": "endothelial/CAF/immune co-culture, microfluidic model, or immune-competent in vivo model",
            "do_not_use_as_primary": "tumor-cell-only viability screen",
        },
        {
            "strategy": "hybrid_panERBB_mTORC12", "priority": 2,
            "status": "RNA panel exploratory; phosphoprotein confirmation required",
            "selection": "true epithelial-mesenchymal coexpression + ERBB activity + mTORC1/2 activity",
            "intervention": "pan-ErbB plus dual mTORC1/2 inhibition",
            "required_model": "3D tumor organoid with matched non-hybrid controls",
            "do_not_use_as_primary": "bulk RNA coactivation without single-cell/spatial confirmation",
        },
        {
            "strategy": "hybrid_panERBB_AKT", "priority": 3,
            "status": "full-transcriptome hypothesis only; compact RNA selector rejected",
            "selection": "true epithelial-mesenchymal coexpression + ERBB activity + phospho-AKT activity",
            "intervention": "pan-ErbB plus AKT inhibition",
            "required_model": "3D tumor organoid with phosphoprotein-qualified pathway activation",
            "do_not_use_as_primary": "failed compact AKT RNA panel",
        },
    ]
    pd.DataFrame(strategy).to_csv(OUT / "strategy_tiers.tsv", sep="\t", index=False)

    assays = []
    def add(strategy_name, domain, modality, markers, role, required):
        for marker in markers:
            assays.append({
                "strategy": strategy_name, "domain": domain, "modality": modality,
                "marker": marker, "role": role, "required_for_gate": required,
            })

    m = "mesenchymal_antiangiogenic_IO"
    add(m, "mesenchymal_state", "RNA", ["MMP2", "FSTL1", "GAS1", "DPYSL3", "DCN", "COL6A2"], "locked compact panel", True)
    add(m, "vascular_stromal_barrier", "RNA", ["FSTL1", "NRP1", "AKAP12", "DCN", "ACVR1", "LUM", "COL3A1", "LOX"], "locked compact panel", True)
    add(m, "immune_visibility", "RNA", ["CCL5", "B2M", "CXCL10", "CXCL11", "PSMB9", "CCL4", "IDO1", "CXCL9"], "locked compact panel", True)
    add(m, "incomplete_completion", "RNA", ["CXCL1", "CCL2", "CXCL5", "CXCL3", "CCL20", "PDCD1LG2", "CXCL2", "CD86"], "locked compact panel", True)
    add(m, "spatial_context", "multiplex_IF_IHC", ["CD8", "CD68", "FOXP3", "PD-L1", "panCK", "COL1", "CD31"], "localize immune exclusion and vascular/stromal context", True)
    add(m, "functional_PD", "functional", ["T-cell infiltration", "tumor-cell killing", "endothelial permeability", "vessel normalization", "hypoxia"], "mechanism readout", True)

    for h in ["hybrid_panERBB_mTORC12", "hybrid_panERBB_AKT"]:
        add(h, "hybrid_state", "single_cell_or_spatial", ["EPCAM", "KRT8", "KRT18", "VIM", "ZEB1", "FN1", "ITGA5"], "prove same-cell E/M coexpression", True)
        add(h, "ERBB_dependency", "protein_or_genomic", ["EGFR", "HER2", "HER3", "PTK6", "pEGFR", "pHER2", "pHER3", "ERBB2_amplification"], "confirm target/pathway activity", True)
    add("hybrid_panERBB_mTORC12", "mTOR_activity", "phosphoprotein", ["pS6", "p4EBP1", "pAKT_S473"], "confirm mTORC1/2 activity and target engagement", True)
    add("hybrid_panERBB_AKT", "AKT_activity", "phosphoprotein", ["pAKT_T308", "pAKT_S473", "pPRAS40", "pGSK3B"], "replace rejected compact RNA selector", True)
    pd.DataFrame(assays).drop_duplicates().to_csv(OUT / "multimodal_assay_panel.tsv", sep="\t", index=False)

    experiments = [
        ["H1", "hybrid_panERBB_AKT", "3D organoid", "pan-ErbB inhibitor", "AKT inhibitor", "6x6 dose matrix", "72-120h", "Bliss; Loewe; ZIP; HSA", "viability; apoptosis", "24h phosphoprotein target engagement"],
        ["H2", "hybrid_panERBB_mTORC12", "3D organoid", "pan-ErbB inhibitor", "dual mTORC1/2 inhibitor", "6x6 dose matrix", "72-120h", "Bliss; Loewe; ZIP; HSA", "viability; apoptosis", "24h pS6/p4EBP1 suppression"],
        ["M1", "mesenchymal_antiangiogenic_IO", "endothelial+CAF+tumor+immune co-culture", "low-dose FGFR/VEGFR inhibitor", "PD-1/PD-L1 blockade", "factorial single agents+combination", "24h;72h;7d", "interaction contrast; not tumor-only synergy", "T-cell infiltration and tumor killing", "permeability; CD31; hypoxia; ECM"],
        ["M2", "mesenchymal_antiangiogenic_IO", "immune-competent in vivo", "low-dose FGFR/VEGFR inhibitor", "PD-1/PD-L1 blockade", "2x2 randomized factorial", "longitudinal", "treatment interaction with CI", "tumor growth and immune infiltration", "toxicity; vascular normalization; spatial immune distance"],
    ]
    columns = ["experiment_id", "strategy", "model", "agent_A_class", "agent_B_class", "design", "timepoints", "combination_analysis", "primary_endpoint", "secondary_mechanism_endpoints"]
    pd.DataFrame(experiments, columns=columns).to_csv(OUT / "experimental_matrix.tsv", sep="\t", index=False)

    gates = [
        [1, "model qualification", "biomarker-high and biomarker-low/nonmatching controls represented; hybrid identity proven in the same cells", "stop or replace model"],
        [2, "single-agent calibration", "non-saturating exposures with measurable target engagement and acceptable viability window", "recalibrate dose range"],
        [3, "combination screen", "positive interaction direction in at least 3 synergy/reference metrics and replicated in independent biological repeats", "do not advance combination"],
        [4, "mechanism", "combination changes the prespecified pathway/ecosystem endpoint beyond both single agents", "classify as non-mechanistic"],
        [5, "biomarker specificity", "effect stronger in qualified state-high models than state-low/nonmatching controls", "do not claim state specificity"],
        [6, "independent validation", "same direction in an independent model system with uncertainty intervals excluding a null-sized effect chosen before confirmation", "retain as hypothesis only"],
    ]
    pd.DataFrame(gates, columns=["gate", "stage", "advance_rule", "failure_action"]).to_csv(OUT / "stage_gates.tsv", sep="\t", index=False)

    rigor = [
        ["allocation", "randomize wells/animals after model qualification"],
        ["blinding", "blind endpoint quantification and spatial image analysis"],
        ["replication", "at least 3 independent biological repeats for screens; technical replicates do not replace biological repeats"],
        ["sample_size", "estimate confirmatory n from pilot variance and a prespecified minimum biologically relevant interaction; do not power from observed pilot p value alone"],
        ["multiplicity", "one primary endpoint per experiment; adjust within each prespecified family of secondary endpoints"],
        ["exclusions", "freeze QC/exclusion rules before unblinding treatment groups"],
        ["reporting", "report all single agents, combination, uncertainty intervals, toxicity and failed models"],
    ]
    pd.DataFrame(rigor, columns=["item", "frozen_rule"]).to_csv(OUT / "experimental_rigor_lock.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "prospective multimodal assay and experimental validation blueprint",
        "priority_order": [x["strategy"] for x in sorted(strategy, key=lambda x: x["priority"])],
        "compact_RNA_panel_accepted": ["mesenchymal_antiangiogenic_IO"],
        "compact_RNA_panel_exploratory": ["hybrid_panERBB_mTORC12"],
        "compact_RNA_panel_rejected": ["hybrid_panERBB_AKT"],
        "key_correction": "AKT/mTOR activity requires phosphoprotein confirmation; hybrid identity requires same-cell or spatial confirmation",
        "drug_names_frozen": False,
        "treatment_biomarker_validated": False,
        "combination_efficacy_claimed": False,
        "next_gate": "inventory available models/reagents and freeze exact compounds, doses, controls and confirmatory sample-size inputs",
    }
    (OUT / "prospective_validation_blueprint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
