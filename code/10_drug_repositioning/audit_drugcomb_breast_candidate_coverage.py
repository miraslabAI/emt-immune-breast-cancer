#!/usr/bin/env python
"""Outcome-blind coverage audit for state-specific combination hypotheses in DrugComb v1.5.

The script reads the 1.4-GB table in chunks, removes monotherapy rows, links
breast cell lines to the frozen EMT-state manifest, and reports whether proposed
exact drugs or mechanistic classes were actually tested.  It does not select on
synergy scores and does not fit an efficacy model.
"""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "02_data/01_raw/drug_combination/drugcomb_v1_5/summary_v_1_5.csv"
STATES = ROOT / "04_analysis/11_drug_repositioning/10_pharmacogenomic_validation_lock/frozen_ccle_breast_emt_states.tsv"
HYPOTHESES = ROOT / "04_analysis/11_drug_repositioning/14_integrated_state_specific_combination_matrix/state_specific_combination_hypotheses.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/15_drugcomb_breast_coverage_audit"
EXPECTED_MD5 = "cd3e24758c10ffaea0124b6e80626c32"
CHUNK_SIZE = 200_000
SYNERGY = ["synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
USECOLS = [
    "block_id", "drug_row", "drug_col", "cell_line_name", "study_name", "tissue_name",
    "css_ri", *SYNERGY, "drug_row_clinical_phase", "drug_col_clinical_phase",
    "drug_row_target_name", "drug_col_target_name",
]


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def build_cell_aliases(states: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    mapping: dict[str, dict[str, object]] = {}
    for _, row in states.iterrows():
        aliases = {
            str(row["StrippedCellLineName"]),
            str(row["CCLEName"]),
            str(row["CCLEName"]).split("_")[0],
        }
        for alias in aliases:
            key = norm(alias)
            if not key:
                continue
            payload = {
                "ModelID": row["ModelID"],
                "CCLEName": row["CCLEName"],
                "StrippedCellLineName": row["StrippedCellLineName"],
                "emt_state": row["emt_state"],
                "epithelial_core": row["epithelial_core"],
                "mesenchymal_core": row["mesenchymal_core"],
                "emt_position": row["emt_position"],
            }
            if key in mapping and mapping[key]["ModelID"] != payload["ModelID"]:
                raise ValueError(f"Ambiguous frozen cell-line alias: {alias}")
            mapping[key] = payload
            records.append({"normalized_alias": key, "alias": alias, **payload})
    return pd.DataFrame(records).drop_duplicates(), mapping


DRUG_GROUPS = {
    "hybrid_erbb_discovery": {
        "BMS-690514", "poziotinib", "AZD8931", "sapitinib", "dacomitinib",
        "vandetanib", "XL-647", "ARRY-334543",
    },
    "hybrid_erbb_clinical_analogues": {
        "afatinib", "lapatinib", "neratinib", "tucatinib", "pyrotinib",
    },
    "hybrid_rock_partner": {
        "belumosudil", "belumosudil mesylate", "fasudil", "ripasudil",
        "netarsudil", "netarsudil dimesylate",
    },
    "mesenchymal_brivanib": {"brivanib", "brivanib alaninate"},
    "mesenchymal_fgf_vegf_analogues": {
        "axitinib", "cediranib", "erdafitinib", "futibatinib", "lenvatinib",
        "nintedanib", "pazopanib", "regorafenib", "sorafenib", "sunitinib",
        "tivozanib",
    },
    "mesenchymal_cox2_partner": {"celecoxib"},
}
DRUG_GROUPS_NORM = {key: {norm(x) for x in values} for key, values in DRUG_GROUPS.items()}


PAIR_DEFINITIONS = [
    {
        "hypothesis_id": "H_ERBB_ROCK_EXACT",
        "state": "hybrid-peak",
        "anchor_groups": ["hybrid_erbb_discovery"],
        "partner_groups": ["hybrid_rock_partner"],
        "coverage_level": "exact_or_named_class",
        "cell_line_assay_evaluable": True,
    },
    {
        "hypothesis_id": "H_ERBB_ROCK_EXPANDED",
        "state": "hybrid-peak",
        "anchor_groups": ["hybrid_erbb_discovery", "hybrid_erbb_clinical_analogues"],
        "partner_groups": ["hybrid_rock_partner"],
        "coverage_level": "mechanistic_class",
        "cell_line_assay_evaluable": True,
    },
    {
        "hypothesis_id": "M_BRIVANIB_COX2_EXACT",
        "state": "mesenchymal-enriched",
        "anchor_groups": ["mesenchymal_brivanib"],
        "partner_groups": ["mesenchymal_cox2_partner"],
        "coverage_level": "exact_or_named_class",
        "cell_line_assay_evaluable": True,
    },
    {
        "hypothesis_id": "M_ANTIANGIOGENIC_COX2_EXPANDED",
        "state": "mesenchymal-enriched",
        "anchor_groups": ["mesenchymal_brivanib", "mesenchymal_fgf_vegf_analogues"],
        "partner_groups": ["mesenchymal_cox2_partner"],
        "coverage_level": "mechanistic_class",
        "cell_line_assay_evaluable": True,
    },
    {
        "hypothesis_id": "H_ERBB_CHECKPOINT",
        "state": "hybrid-peak",
        "anchor_groups": ["hybrid_erbb_discovery", "hybrid_erbb_clinical_analogues"],
        "partner_groups": [],
        "coverage_level": "immune_partner",
        "cell_line_assay_evaluable": False,
    },
    {
        "hypothesis_id": "M_BRIVANIB_CHECKPOINT",
        "state": "mesenchymal-enriched",
        "anchor_groups": ["mesenchymal_brivanib"],
        "partner_groups": [],
        "coverage_level": "immune_partner",
        "cell_line_assay_evaluable": False,
    },
]


def group_membership(drug_norm: pd.Series, groups: list[str]) -> pd.Series:
    allowed: set[str] = set()
    for group in groups:
        allowed |= DRUG_GROUPS_NORM[group]
    return drug_norm.isin(allowed)


def pair_mask(frame: pd.DataFrame, definition: dict[str, object]) -> pd.Series:
    anchor_row = group_membership(frame["drug_row_norm"], definition["anchor_groups"])
    anchor_col = group_membership(frame["drug_col_norm"], definition["anchor_groups"])
    partner_row = group_membership(frame["drug_row_norm"], definition["partner_groups"])
    partner_col = group_membership(frame["drug_col_norm"], definition["partner_groups"])
    return (anchor_row & partner_col) | (anchor_col & partner_row)


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    if RAW.stat().st_size != 1_421_694_989:
        raise ValueError(f"Unexpected DrugComb file size: {RAW.stat().st_size}")
    OUT.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(STATES, sep="\t", low_memory=False)
    hypotheses = pd.read_csv(HYPOTHESES, sep="\t", low_memory=False)
    aliases, alias_map = build_cell_aliases(states)
    aliases.to_csv(OUT / "frozen_breast_cell_line_aliases.tsv", sep="\t", index=False)

    breast_blocks: list[pd.DataFrame] = []
    candidate_blocks: list[pd.DataFrame] = []
    raw_rows = combo_rows = breast_tissue_rows = linked_rows = 0
    seen_raw_cell_lines: set[str] = set()
    seen_linked_models: set[str] = set()
    studies: set[str] = set()

    for chunk in pd.read_csv(
        RAW,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        na_values=[r"\N"],
    ):
        raw_rows += len(chunk)
        chunk["drug_row"] = chunk["drug_row"].fillna("").astype(str).str.strip()
        chunk["drug_col"] = chunk["drug_col"].fillna("").astype(str).str.strip()
        chunk["drug_row_norm"] = chunk["drug_row"].map(norm)
        chunk["drug_col_norm"] = chunk["drug_col"].map(norm)
        chunk["cell_line_norm"] = chunk["cell_line_name"].map(norm)

        is_combo = ~chunk["drug_col_norm"].isin({"", "null", "na", "none"})
        chunk = chunk[is_combo].copy()
        combo_rows += len(chunk)
        if chunk.empty:
            continue

        tissue_breast = chunk["tissue_name"].fillna("").astype(str).str.casefold().str.contains("breast")
        linked = chunk["cell_line_norm"].isin(alias_map)
        breast_tissue_rows += int(tissue_breast.sum())
        keep = tissue_breast | linked
        chunk = chunk[keep].copy()
        if chunk.empty:
            continue

        seen_raw_cell_lines.update(chunk["cell_line_name"].dropna().astype(str))
        chunk["linked_to_frozen_state"] = chunk["cell_line_norm"].isin(alias_map)
        for column in ["ModelID", "CCLEName", "StrippedCellLineName", "emt_state", "epithelial_core", "mesenchymal_core", "emt_position"]:
            chunk[column] = chunk["cell_line_norm"].map(
                lambda key, col=column: alias_map.get(key, {}).get(col, np.nan)
            )
        linked_rows += int(chunk["linked_to_frozen_state"].sum())
        seen_linked_models.update(chunk.loc[chunk["linked_to_frozen_state"], "ModelID"].astype(str))
        studies.update(chunk["study_name"].dropna().astype(str))
        for column in ["css_ri", *SYNERGY]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        breast_blocks.append(chunk)

        all_candidate_names = set().union(*DRUG_GROUPS_NORM.values())
        candidate = chunk[
            chunk["drug_row_norm"].isin(all_candidate_names)
            | chunk["drug_col_norm"].isin(all_candidate_names)
        ]
        if len(candidate):
            candidate_blocks.append(candidate)

    if not breast_blocks:
        raise ValueError("No breast or frozen-model combination rows were found")
    breast = pd.concat(breast_blocks, ignore_index=True)
    candidates = pd.concat(candidate_blocks, ignore_index=True) if candidate_blocks else breast.iloc[0:0].copy()

    breast.to_csv(OUT / "drugcomb_breast_combination_blocks.tsv.gz", sep="\t", index=False, compression="gzip")
    candidates.to_csv(OUT / "candidate_involving_combination_blocks.tsv.gz", sep="\t", index=False, compression="gzip")

    linkage = (
        breast.groupby(["cell_line_name", "cell_line_norm", "linked_to_frozen_state", "ModelID", "emt_state"], dropna=False)
        .agg(blocks=("block_id", "nunique"), studies=("study_name", "nunique"))
        .reset_index()
        .sort_values(["linked_to_frozen_state", "blocks"], ascending=[False, False])
    )
    linkage.to_csv(OUT / "drugcomb_breast_cell_line_linkage.tsv", sep="\t", index=False)

    coverage_rows: list[dict[str, object]] = []
    matched_pair_rows: list[pd.DataFrame] = []
    for definition in PAIR_DEFINITIONS:
        if not definition["cell_line_assay_evaluable"]:
            coverage_rows.append({
                **definition,
                "observed_blocks": 0,
                "observed_linked_blocks": 0,
                "observed_cell_lines": 0,
                "observed_linked_models": 0,
                "observed_studies": 0,
                "synergy_nonmissing_blocks": 0,
                "coverage_decision": "NOT_EVALUABLE_IN_TUMOR_CELL_ONLY_DRUGCOMB",
            })
            continue
        mask = pair_mask(breast, definition)
        hit = breast[mask].copy()
        if len(hit):
            hit["hypothesis_id"] = definition["hypothesis_id"]
            matched_pair_rows.append(hit)
        synergy_nonmissing = hit[SYNERGY].notna().any(axis=1) if len(hit) else pd.Series(dtype=bool)
        linked_hit = hit[hit["linked_to_frozen_state"]] if len(hit) else hit
        coverage_rows.append({
            **definition,
            "observed_blocks": int(hit["block_id"].nunique()),
            "observed_linked_blocks": int(linked_hit["block_id"].nunique()),
            "observed_cell_lines": int(hit["cell_line_name"].nunique()),
            "observed_linked_models": int(linked_hit["ModelID"].nunique()),
            "observed_studies": int(hit["study_name"].nunique()),
            "synergy_nonmissing_blocks": int(synergy_nonmissing.sum()),
            "coverage_decision": "DIRECT_PAIR_DATA_AVAILABLE" if len(linked_hit) else (
                "PAIR_DATA_ONLY_OUTSIDE_FROZEN_MODELS" if len(hit) else "PAIR_NOT_OBSERVED"
            ),
        })

    coverage = pd.DataFrame(coverage_rows)
    for column in ["anchor_groups", "partner_groups"]:
        coverage[column] = coverage[column].map(lambda x: ";".join(x))
    coverage.to_csv(OUT / "state_combination_hypothesis_coverage.tsv", sep="\t", index=False)

    if matched_pair_rows:
        pd.concat(matched_pair_rows, ignore_index=True).to_csv(
            OUT / "observed_hypothesis_pair_blocks.tsv", sep="\t", index=False
        )
    else:
        breast.iloc[0:0].assign(hypothesis_id=pd.Series(dtype=str)).to_csv(
            OUT / "observed_hypothesis_pair_blocks.tsv", sep="\t", index=False
        )

    candidate_counts = []
    for group, names in DRUG_GROUPS_NORM.items():
        mask = breast["drug_row_norm"].isin(names) | breast["drug_col_norm"].isin(names)
        hit = breast[mask]
        candidate_counts.append({
            "drug_group": group,
            "named_members": ";".join(sorted(DRUG_GROUPS[group])),
            "combination_blocks": int(hit["block_id"].nunique()),
            "linked_blocks": int(hit.loc[hit["linked_to_frozen_state"], "block_id"].nunique()),
            "cell_lines": int(hit["cell_line_name"].nunique()),
            "linked_models": int(hit.loc[hit["linked_to_frozen_state"], "ModelID"].nunique()),
            "studies": int(hit["study_name"].nunique()),
        })
    pd.DataFrame(candidate_counts).to_csv(OUT / "candidate_group_combination_coverage.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind DrugComb breast/candidate coverage audit; no synergy hypothesis test",
        "source_file_size_bytes": RAW.stat().st_size,
        "source_expected_md5": EXPECTED_MD5,
        "raw_rows": int(raw_rows),
        "combination_rows_after_monotherapy_exclusion": int(combo_rows),
        "breast_tissue_rows": int(breast_tissue_rows),
        "retained_breast_or_frozen_model_rows": int(len(breast)),
        "linked_rows": int(linked_rows),
        "raw_breast_cell_lines": int(len(seen_raw_cell_lines)),
        "linked_frozen_models": int(len(seen_linked_models)),
        "frozen_models_total": int(states["ModelID"].nunique()),
        "studies": sorted(studies),
        "hypothesis_coverage_decisions": dict(zip(coverage["hypothesis_id"], coverage["coverage_decision"])),
        "immune_pair_constraint": "PD-1/PD-L1 combinations cannot be validated in tumor-cell-only viability screens",
        "next_gate": "freeze only observed state-linked pairs for study-aware synergy testing; retain unobserved pairs as mechanistic predictions",
    }
    (OUT / "drugcomb_breast_coverage_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
