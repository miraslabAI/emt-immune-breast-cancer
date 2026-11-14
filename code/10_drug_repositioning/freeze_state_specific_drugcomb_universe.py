#!/usr/bin/env python
"""Freeze an outcome-blind DrugComb universe for EMT-state partner discovery."""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
BLOCKS = BASE / "15_drugcomb_breast_coverage_audit/drugcomb_breast_combination_blocks.tsv.gz"
OUT = BASE / "16_state_specific_drugcomb_lock"
DETERMINATE = {"epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"}
SYNERGY = ["synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
MIN_TARGET_MODELS = 3
MIN_OTHER_MODELS = 3


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


FAMILIES = {
    "hybrid_erbb_partner_discovery": {
        "state": "hybrid-peak",
        "anchors": {
            "BMS-690514", "poziotinib", "AZD8931", "sapitinib", "dacomitinib",
            "vandetanib", "XL-647", "ARRY-334543", "afatinib", "lapatinib",
            "neratinib", "tucatinib", "pyrotinib",
        },
    },
    "mesenchymal_antiangiogenic_partner_discovery": {
        "state": "mesenchymal-enriched",
        "anchors": {
            "brivanib", "brivanib alaninate", "axitinib", "cediranib", "erdafitinib",
            "futibatinib", "lenvatinib", "nintedanib", "pazopanib", "regorafenib",
            "sorafenib", "sunitinib", "tivozanib",
        },
    },
}


def orient_pairs(frame: pd.DataFrame, family: str, state: str, anchors: set[str]) -> pd.DataFrame:
    anchors_norm = {norm(x) for x in anchors}
    row_anchor = frame["drug_row_norm"].isin(anchors_norm)
    col_anchor = frame["drug_col_norm"].isin(anchors_norm)
    selected = frame[row_anchor | col_anchor].copy()
    selected["anchor"] = selected["drug_row"].where(row_anchor.loc[selected.index], selected["drug_col"])
    selected["partner"] = selected["drug_col"].where(row_anchor.loc[selected.index], selected["drug_row"])
    selected["anchor_norm"] = selected["anchor"].map(norm)
    selected["partner_norm"] = selected["partner"].map(norm)
    selected = selected[selected["anchor_norm"].ne(selected["partner_norm"])].copy()
    selected = selected[~selected["partner_norm"].isin(anchors_norm)].copy()
    selected["analysis_family"] = family
    selected["target_state"] = state
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BLOCKS, sep="\t", low_memory=False)
    frame = frame[
        frame["linked_to_frozen_state"].astype(str).str.casefold().eq("true")
        & frame["emt_state"].isin(DETERMINATE)
    ].copy()
    frame["drug_row_norm"] = frame["drug_row"].map(norm)
    frame["drug_col_norm"] = frame["drug_col"].map(norm)
    for column in [*SYNERGY, "css_ri", "emt_position"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame[SYNERGY].notna().any(axis=1)].copy()

    oriented = []
    for family, definition in FAMILIES.items():
        oriented.append(orient_pairs(frame, family, definition["state"], definition["anchors"]))
    candidates = pd.concat(oriented, ignore_index=True)

    lock_rows = []
    locked_blocks = []
    group_keys = ["analysis_family", "target_state", "study_name", "anchor_norm", "partner_norm"]
    for keys, group in candidates.groupby(group_keys, sort=True):
        family, target_state, study, anchor_norm, partner_norm = keys
        target = group[group["emt_state"].eq(target_state)]
        other = group[~group["emt_state"].eq(target_state)]
        n_target = target["ModelID"].nunique()
        n_other = other["ModelID"].nunique()
        if n_target < MIN_TARGET_MODELS or n_other < MIN_OTHER_MODELS:
            continue
        pair_id = "|".join(map(str, keys))
        group = group.copy()
        group["locked_pair_id"] = pair_id
        group["is_target_state"] = group["emt_state"].eq(target_state)
        group["prespecified_partner"] = group["partner_norm"].isin({"celecoxib"})
        locked_blocks.append(group)
        lock_rows.append({
            "locked_pair_id": pair_id,
            "analysis_family": family,
            "target_state": target_state,
            "study_name": study,
            "anchor": ";".join(sorted(group["anchor"].astype(str).unique())),
            "anchor_norm": anchor_norm,
            "partner": ";".join(sorted(group["partner"].astype(str).unique())),
            "partner_norm": partner_norm,
            "target_models": int(n_target),
            "other_models": int(n_other),
            "target_blocks": int(target["block_id"].nunique()),
            "other_blocks": int(other["block_id"].nunique()),
            "prespecified_partner": bool(partner_norm == "celecoxib"),
            "low_power_minimum_coverage": bool(min(n_target, n_other) == 3),
        })

    if not lock_rows:
        raise ValueError("No DrugComb pair passed the frozen model-coverage gate")
    lock = pd.DataFrame(lock_rows).sort_values(
        ["analysis_family", "prespecified_partner", "target_models", "other_models"],
        ascending=[True, False, False, False],
    )
    blocks = pd.concat(locked_blocks, ignore_index=True)
    lock.to_csv(OUT / "frozen_state_specific_drugcomb_pair_universe.tsv", sep="\t", index=False)
    blocks.to_csv(OUT / "frozen_state_specific_drugcomb_blocks.tsv.gz", sep="\t", index=False, compression="gzip")

    counts = (
        lock.groupby(["analysis_family", "study_name", "prespecified_partner"], dropna=False)
        .agg(
            locked_pairs=("locked_pair_id", "nunique"),
            anchors=("anchor_norm", "nunique"),
            partners=("partner_norm", "nunique"),
        )
        .reset_index()
    )
    counts.to_csv(OUT / "frozen_drugcomb_universe_counts.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind state-specific DrugComb test-universe freeze",
        "minimum_target_models": MIN_TARGET_MODELS,
        "minimum_other_models": MIN_OTHER_MODELS,
        "locked_pairs": int(lock["locked_pair_id"].nunique()),
        "locked_blocks": int(blocks["block_id"].nunique()),
        "families": lock.groupby("analysis_family")["locked_pair_id"].nunique().to_dict(),
        "prespecified_celecoxib_pairs": int(lock["prespecified_partner"].sum()),
        "outcomes_used_for_selection": False,
        "immune_pairs_included": False,
        "next_gate": "test frozen pairs using consensus synergy as primary endpoint and four individual synergy models as secondary endpoints",
    }
    (OUT / "state_specific_drugcomb_lock_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
