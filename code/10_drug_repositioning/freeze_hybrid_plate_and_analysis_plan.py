#!/usr/bin/env python
"""Freeze practical Hybrid models, 96-well layouts, and combination-analysis gates."""
from __future__ import annotations

import os

import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK = ROOT / "04_analysis/11_drug_repositioning/27_hybrid_compound_model_lock"
OUT = ROOT / "04_analysis/11_drug_repositioning/28_hybrid_plate_analysis_lock"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shortlist = pd.read_csv(LOCK / "locked_hybrid_model_shortlist.tsv", sep="\t")
    model_plan = [
        ["HCC1954", "core_primary", 1, "HER2+ Hybrid reference", "ATCC CRL-2338", "exact published sapitinib-capivasertib xenograft model"],
        ["MDAMB361", "core_primary", 2, "ER+/HER2+ high coactivation", "ATCC HTB-27", "lineage-diverse Hybrid model"],
        ["BT20", "core_primary", 3, "TNBC high ERBB/AKT proxy", "ATCC HTB-19", "tests beyond HER2-amplified context"],
        ["HCC70", "core_primary", 4, "TNBC Hybrid", "ATCC CRL-2315", "tests beyond HER2-amplified context"],
        ["21NT", "extension_if_accessible", 1, "HER2+ high coactivation", "DFCI/source MTA", "strong same-lineage Hybrid extension"],
        ["21MT2", "extension_if_accessible", 2, "HER2+ high coactivation", "DFCI/source MTA", "strong same-lineage Hybrid extension"],
        ["CAL851", "state_anchor_control", 1, "Hybrid/ERBB-low", "verify supplier", "separates Hybrid identity from ERBB dependence"],
        ["AU565", "state_anchor_control", 2, "non-Hybrid/ERBB-high", "verify repository", "separates ERBB dependence from Hybrid identity"],
        ["UACC893", "backup_control", 1, "non-Hybrid/ERBB-high", "verify repository", "backup for AU565"],
        ["HCC38", "backup_hybrid", 1, "TNBC Hybrid", "verify repository", "backup if a core model fails phosphoprotein qualification"],
        ["HCC1143", "backup_hybrid", 2, "TNBC high coactivation", "ATCC CRL-2321", "backup if a core model fails phosphoprotein qualification"],
    ]
    plan_cols = ["cell_line", "role", "priority_within_role", "biological_role", "source_status", "reason"]
    plan = pd.DataFrame(model_plan, columns=plan_cols)
    plan = plan.merge(shortlist.drop_duplicates("StrippedCellLineName"), left_on="cell_line", right_on="StrippedCellLineName", how="left")
    if plan["ModelID"].isna().any():
        raise ValueError(f"Model lock linkage failed: {plan.loc[plan.ModelID.isna(), 'cell_line'].tolist()}")
    plan.to_csv(OUT / "primary_backup_model_lock.tsv", sep="\t", index=False)

    pairs = pd.DataFrame([
        ["P1", "sapitinib", "capivasertib", "primary_reference"],
        ["P2", "neratinib", "capivasertib", "primary_translational_class_test"],
    ], columns=["pair_id", "agent_A", "agent_B", "role"])
    pairs.to_csv(OUT / "primary_pair_lock.tsv", sep="\t", index=False)

    # Two identical 6x6 matrices per 96-well plate.  Doses are normalized to the
    # model-specific IC50 established in an earlier calibration experiment.
    levels = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for replicate, start_col in [(1, 1), (2, 7)]:
        for r, a in enumerate(levels):
            for c, b in enumerate(levels):
                rows.append({"well": f"{'ABCDEF'[r]}{start_col+c}", "technical_replicate": replicate,
                             "well_role": "dose_matrix", "agent_A_IC50_multiple": a,
                             "agent_B_IC50_multiple": b})
    controls = [
        ("G", range(1, 4), "media_blank"), ("G", range(4, 7), "vehicle_control"),
        ("G", range(7, 10), "positive_kill_control"), ("G", range(10, 13), "no_cell_blank"),
        ("H", range(1, 7), "vehicle_control"), ("H", range(7, 13), "spare_prespecified_QC_only"),
    ]
    for row, columns, role in controls:
        for col in columns:
            rows.append({"well": f"{row}{col}", "technical_replicate": 0, "well_role": role,
                         "agent_A_IC50_multiple": 0.0, "agent_B_IC50_multiple": 0.0})
    layout = pd.DataFrame(rows)
    if len(layout) != 96 or layout["well"].duplicated().any(): raise ValueError("Invalid 96-well layout")
    layout.to_csv(OUT / "locked_96well_layout.tsv", sep="\t", index=False)

    screening_models = plan[plan["role"].isin(["core_primary", "state_anchor_control"])]["cell_line"].tolist()
    manifest = []
    plate = 0
    for biological_repeat, model, pair_id in itertools.product(range(1, 4), screening_models, pairs["pair_id"]):
        plate += 1
        manifest.append({"plate_id": f"HYB-{plate:03d}", "biological_repeat": biological_repeat,
                         "cell_line": model, "pair_id": pair_id, "layout": "locked_96well_layout.tsv",
                         "randomized_processing_order": "assign with seeded randomization immediately before experiment"})
    pd.DataFrame(manifest).to_csv(OUT / "primary_plate_manifest.tsv", sep="\t", index=False)

    analysis = [
        [1, "plate_QC", "blank-subtracted vehicle CV <=20% and Z-prime >=0.5 using positive-kill and vehicle controls", "exclude plate before viewing combination scores"],
        [2, "normalization", "normalize to within-plate vehicle median after blank subtraction; retain raw luminescence", "no cross-plate rescaling by treatment response"],
        [3, "primary_estimand", "mean Bliss excess across the complete 5x5 non-zero combination grid, averaged across technical replicates", "do not select the maximum local synergy cell"],
        [4, "secondary_estimands", "ZIP, Loewe and HSA over the same locked grid", "supportive; no substitution for failed primary estimand"],
        [5, "model_positive", "median Bliss score >10, bootstrap 95% CI above 0, and positive direction in all 3 biological repeats", "all conditions required"],
        [6, "pair_advance", "positive in >=3 of 4 core Hybrid models, including >=1 HER2+ and >=1 TNBC model", "controls are not counted toward numerator"],
        [7, "state_specificity", "interaction contrast stronger in core Hybrid models than both Hybrid/ERBB-low and non-Hybrid/ERBB-high controls", "report effect and CI; do not infer from separate p values"],
        [8, "mechanism", "combined suppression of pERBB feedback and pPRAS40/pS6 plus apoptosis beyond both single agents", "required before declaring mechanistic combination support"],
        [9, "multiplicity", "Holm adjustment across the two primary pairs for the pair-level primary test", "secondary metrics remain labeled secondary"],
    ]
    pd.DataFrame(analysis, columns=["order", "item", "frozen_rule", "failure_action_or_note"]).to_csv(
        OUT / "combination_analysis_lock.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS", "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "practical model, plate-layout and combination-analysis lock",
        "core_models": plan.loc[plan.role.eq("core_primary"), "cell_line"].tolist(),
        "state_anchor_controls": plan.loc[plan.role.eq("state_anchor_control"), "cell_line"].tolist(),
        "conditional_extensions": plan.loc[plan.role.eq("extension_if_accessible"), "cell_line"].tolist(),
        "primary_pairs": pairs[["agent_A", "agent_B"]].agg(" + ".join, axis=1).tolist(),
        "biological_repeats": 3, "technical_matrix_replicates_per_plate": 2,
        "primary_screen_plates": len(manifest),
        "response_data_used_for_model_selection": False,
        "next_gate": "freeze computational result-ingestion templates and synergy-analysis implementation; wet-lab execution remains external",
    }
    (OUT / "hybrid_plate_analysis_lock_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
