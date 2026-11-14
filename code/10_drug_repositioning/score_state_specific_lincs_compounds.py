#!/usr/bin/env python
"""Score breast-cell LINCS perturbations against frozen EMT-state objectives.

The Level-5 matrix is read in bounded chunks.  Compounds are rewarded for
decreasing frozen state-specific target modules and penalized for suppressing
immune-visibility preservation modules.  Epithelial results remain
exploratory until independent viability/dependency support is added.
"""
from __future__ import annotations

import os

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK_DIR = ROOT / "04_analysis/11_drug_repositioning/08_state_specific_lincs_query_lock"
LOCK_SUMMARY = LOCK_DIR / "state_specific_lincs_query_lock_summary.json"
MODULE_FILE = LOCK_DIR / "frozen_state_lincs_gene_modules.tsv"
SIGNATURE_FILE = LOCK_DIR / "eligible_breast_lincs_signatures.tsv.gz"
MECHANISM_MATCHES = LOCK_DIR / "direction_compatible_chembl_lincs_name_matches.tsv"
LINCS_DIR = ROOT / "02_data/01_raw/reference/lincs/GSE70138"
GENE_INFO = LINCS_DIR / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
LEVEL5 = LINCS_DIR / "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
OUT = ROOT / "04_analysis/11_drug_repositioning/09_state_specific_lincs_scoring"

PRESERVATION_MARGIN = 0.25
PRESERVATION_GATE = -0.50
PRIMARY_TIME_H = 24.0
MIN_CELLS_FOR_RANKING = 2


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def robust_z(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    finite = np.isfinite(array)
    result = np.full(len(array), np.nan, dtype=float)
    if finite.sum() < 5:
        return pd.Series(result, index=values.index)
    x = array[finite]
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if not np.isfinite(scale) or scale < 1e-8:
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(x, ddof=1))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    result[finite] = (x - center) / scale
    return pd.Series(np.clip(result, -8.0, 8.0), index=values.index)


def semijoin_modules(modules: pd.DataFrame, row_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = pd.read_csv(GENE_INFO, sep="\t", compression="gzip", dtype=str)
    genes["gene"] = genes["pr_gene_symbol"].astype(str).str.upper()
    genes["gene_id"] = genes["pr_gene_id"].astype(str)
    genes = genes[
        genes["gene_id"].isin(row_ids)
        & pd.to_numeric(genes["pr_is_bing"], errors="coerce").eq(1)
    ].drop_duplicates("gene", keep="first")
    mapped = modules[modules["lincs_bing"].astype(str).str.casefold().eq("true")].merge(
        genes[["gene", "gene_id"]], on="gene", how="inner", validate="many_to_one"
    )
    requested = modules.groupby(["state", "module"], as_index=False).agg(requested_genes=("gene", "nunique"))
    covered = mapped.groupby(["state", "module"], as_index=False).agg(
        level5_genes=("gene", "nunique"),
        level5_gene_ids=("gene_id", "nunique"),
    )
    coverage = requested.merge(covered, on=["state", "module"], how="left").fillna(0)
    coverage["level5_coverage"] = coverage["level5_genes"] / coverage["requested_genes"]
    if (coverage["level5_genes"] < 5).any():
        bad = coverage.loc[coverage["level5_genes"] < 5, ["state", "module", "level5_genes"]]
        raise RuntimeError(f"Frozen module lost Level-5 coverage: {bad.to_dict('records')}")
    return mapped, coverage


def read_module_scores(signatures: pd.DataFrame, modules: pd.DataFrame,
                       chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(LEVEL5, "r") as handle:
        col_ids = decode(handle["0/META/COL/id"][:])
        row_ids = decode(handle["0/META/ROW/id"][:])
        sig_lookup = {value: index for index, value in enumerate(col_ids)}
        missing = sorted(set(signatures["sig_id"].astype(str)) - set(sig_lookup))
        if missing:
            raise ValueError(f"Eligible signatures absent from Level-5: {missing[:10]}")
        mapped, coverage = semijoin_modules(modules, set(row_ids))
        row_lookup = {value: index for index, value in enumerate(row_ids)}

        module_keys = list(mapped.groupby(["state", "module"], sort=True).groups)
        gene_positions = {
            key: np.array(sorted({row_lookup[g] for g in group["gene_id"]}), dtype=int)
            for key, group in mapped.groupby(["state", "module"], sort=True)
        }
        union_positions = np.array(sorted(set().union(*(set(x) for x in gene_positions.values()))), dtype=int)
        union_lookup = {position: index for index, position in enumerate(union_positions)}
        local_positions = {
            key: np.array([union_lookup[x] for x in positions], dtype=int)
            for key, positions in gene_positions.items()
        }

        ordered = sorted(
            ((sig_lookup[sig_id], sig_id) for sig_id in signatures["sig_id"].astype(str)),
            key=lambda item: item[0],
        )
        matrix = handle["0/DATA/0/matrix"]
        score_blocks = []
        for start in range(0, len(ordered), chunk_size):
            block = ordered[start:start + chunk_size]
            positions = np.array([item[0] for item in block], dtype=int)
            # GCTX is signature x gene. Reading complete rows keeps HDF5 access
            # contiguous; only the frozen gene union is retained in memory.
            values = np.asarray(matrix[positions, :], dtype=np.float32)[:, union_positions]
            values[values == -666] = np.nan
            block_scores = {"sig_id": [item[1] for item in block]}
            for key in module_keys:
                with np.errstate(invalid="ignore"):
                    block_scores[f"{key[0]}::{key[1]}"] = np.nanmean(
                        values[:, local_positions[key]], axis=1
                    )
            score_blocks.append(pd.DataFrame(block_scores))
    scores = pd.concat(score_blocks, ignore_index=True)
    if scores["sig_id"].duplicated().any() or len(scores) != len(signatures):
        raise ValueError("Level-5 score table failed one-to-one signature integrity")
    return scores, coverage


def standardize_within_context(frame: pd.DataFrame, module_columns: list[str]) -> pd.DataFrame:
    standardized = frame.copy()
    for column in module_columns:
        standardized[column] = standardized.groupby(
            ["cell_id", "time_h"], group_keys=False, observed=True
        )[column].transform(robust_z)
    return standardized


def build_signature_state_scores(frame: pd.DataFrame, modules: pd.DataFrame) -> pd.DataFrame:
    output = []
    for state, state_modules in modules.groupby("state", sort=True):
        role_by_module = state_modules.drop_duplicates("module").set_index("module")["module_role"].to_dict()
        target_modules = sorted(m for m, role in role_by_module.items() if role in {"primary_target", "exploratory_target"})
        preserve_modules = sorted(m for m, role in role_by_module.items() if role == "preservation_constraint")
        target_cols = [f"{state}::{module}" for module in target_modules]
        preserve_cols = [f"{state}::{module}" for module in preserve_modules]
        if not target_cols or not preserve_cols:
            raise RuntimeError(f"Incomplete target/preservation objectives for {state}")

        result = frame[[
            "sig_id", "pert_id", "pert_iname", "cell_id", "time_h", "dose_value", "tas"
        ]].copy()
        target_values = frame[target_cols].to_numpy(dtype=float)
        preserve_values = frame[preserve_cols].to_numpy(dtype=float)
        result["state"] = state
        result["target_reversal"] = -np.nanmean(target_values, axis=1)
        result["preservation_support"] = np.nanmean(preserve_values, axis=1)
        result["worst_preservation_module"] = np.nanmin(preserve_values, axis=1)
        result["preservation_penalty"] = np.nanmean(
            np.maximum(0.0, -preserve_values - PRESERVATION_MARGIN), axis=1
        )
        result["composite_benefit"] = result["target_reversal"] - result["preservation_penalty"]
        result["preservation_gate_pass"] = result["worst_preservation_module"].ge(PRESERVATION_GATE)
        result["interpretation"] = (
            "exploratory_requires_dependency_validation"
            if state == "epithelial-enriched" else "state_repair_candidate"
        )
        output.append(result)
    return pd.concat(output, ignore_index=True)


def aggregate_compounds(signature_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = signature_scores.groupby(
        ["state", "pert_id", "pert_iname", "cell_id", "time_h"], as_index=False, observed=True
    ).agg(
        signatures=("sig_id", "nunique"),
        target_reversal=("target_reversal", "median"),
        preservation_support=("preservation_support", "median"),
        worst_preservation_module=("worst_preservation_module", "median"),
        preservation_penalty=("preservation_penalty", "median"),
        composite_benefit=("composite_benefit", "median"),
        median_tas=("tas", "median"),
        min_dose=("dose_value", "min"),
        max_dose=("dose_value", "max"),
    )
    cell["cell_positive"] = cell["composite_benefit"].gt(0)
    cell["cell_preservation_pass"] = cell["worst_preservation_module"].ge(PRESERVATION_GATE)

    compound = cell.groupby(
        ["state", "pert_id", "pert_iname", "time_h"], as_index=False, observed=True
    ).agg(
        n_cell_lines=("cell_id", "nunique"),
        n_signatures=("signatures", "sum"),
        median_target_reversal=("target_reversal", "median"),
        median_preservation_support=("preservation_support", "median"),
        q25_worst_preservation=("worst_preservation_module", lambda x: float(np.quantile(x, 0.25))),
        median_preservation_penalty=("preservation_penalty", "median"),
        median_composite_benefit=("composite_benefit", "median"),
        min_cell_composite=("composite_benefit", "min"),
        cell_direction_consistency=("cell_positive", "mean"),
        cell_preservation_consistency=("cell_preservation_pass", "mean"),
        median_tas=("median_tas", "median"),
    )
    compound["preservation_gate_pass"] = (
        compound["q25_worst_preservation"].ge(PRESERVATION_GATE)
        & compound["cell_preservation_consistency"].ge(0.60)
    )
    breadth = np.sqrt(np.minimum(compound["n_cell_lines"], 5) / 5.0)
    consistency = 0.5 + 0.5 * compound["cell_direction_consistency"]
    compound["evidence_adjusted_score"] = compound["median_composite_benefit"] * breadth * consistency
    compound["ranking_eligible"] = compound["n_cell_lines"].ge(MIN_CELLS_FOR_RANKING)
    compound["primary_time"] = np.isclose(compound["time_h"], PRIMARY_TIME_H)
    return cell, compound


def add_mechanisms(compound: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(MECHANISM_MATCHES, sep="\t", low_memory=False)
    if matches.empty:
        compound["direction_compatible_chembl_match"] = False
        return compound, pd.DataFrame()
    mechanism = matches.groupby(["state", "pert_id"], as_index=False).agg(
        chembl_ids=("molecule_chembl_id", lambda x: ";".join(sorted(set(map(str, x))))),
        mechanism_genes=("gene", lambda x: ";".join(sorted(set(map(str, x))))),
        action_types=("action_type", lambda x: ";".join(sorted(set(map(str, x))))),
        mechanisms=("mechanism_of_action", lambda x: ";".join(sorted(set(map(str, x))))),
        maximum_clinical_phase=("molecule_max_phase", "max"),
        mechanism_rows=("gene", "size"),
    )
    annotated = compound.merge(mechanism, on=["state", "pert_id"], how="left", validate="many_to_one")
    annotated["direction_compatible_chembl_match"] = annotated["chembl_ids"].notna()
    return annotated, mechanism


def assign_evidence_tier(frame: pd.DataFrame) -> pd.Series:
    tier = np.full(len(frame), "insufficient_breadth_or_constraint_failure", dtype=object)
    base = frame["ranking_eligible"] & frame["preservation_gate_pass"]
    tier[base.to_numpy()] = "C_state_direction_only"
    b = base & frame["n_cell_lines"].ge(3) & frame["cell_direction_consistency"].ge(0.60)
    tier[b.to_numpy()] = "B_multicell_direction"
    a = b & frame["direction_compatible_chembl_match"] & frame["cell_direction_consistency"].ge(0.67)
    tier[a.to_numpy()] = "A_multicell_plus_mechanism"
    epithelial = frame["state"].eq("epithelial-enriched") & base
    tier[epithelial.to_numpy()] = "E_exploratory_needs_PRISM_GDSC"
    return pd.Series(tier, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 16:
        raise ValueError("--chunk-size must be at least 16")

    required = [LOCK_SUMMARY, MODULE_FILE, SIGNATURE_FILE, MECHANISM_MATCHES, GENE_INFO, LEVEL5]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    lock = json.loads(LOCK_SUMMARY.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS" or lock.get("lincs_level5_values_loaded_or_used") is not False:
        raise RuntimeError("LINCS query lock did not pass a clean metadata-only preflight")

    modules = pd.read_csv(MODULE_FILE, sep="\t", low_memory=False)
    signatures = pd.read_csv(SIGNATURE_FILE, sep="\t", compression="gzip", low_memory=False)
    signatures["time_h"] = pd.to_numeric(signatures["time_h"], errors="coerce")
    signatures["dose_value"] = pd.to_numeric(signatures["dose_value"], errors="coerce")
    signatures["tas"] = pd.to_numeric(signatures["tas"], errors="coerce")
    if signatures["sig_id"].duplicated().any():
        raise ValueError("Duplicate eligible LINCS signature identifiers")

    print(f"Reading {len(signatures)} frozen LINCS signatures in chunks of {args.chunk_size}")
    raw_scores, coverage = read_module_scores(signatures, modules, args.chunk_size)
    scored = signatures.merge(raw_scores, on="sig_id", how="inner", validate="one_to_one")
    module_columns = [column for column in raw_scores if "::" in column]
    standardized = standardize_within_context(scored, module_columns)
    signature_state = build_signature_state_scores(standardized, modules)
    cell, compound = aggregate_compounds(signature_state)
    compound, mechanisms = add_mechanisms(compound)
    compound["evidence_tier"] = assign_evidence_tier(compound)
    compound["rank_within_state_time"] = compound.groupby(
        ["state", "time_h"], observed=True
    )["evidence_adjusted_score"].rank(method="min", ascending=False)

    primary = compound[
        compound["primary_time"] & compound["ranking_eligible"]
    ].sort_values(["state", "evidence_adjusted_score"], ascending=[True, False]).copy()
    top = primary.groupby("state", as_index=False, group_keys=False).head(100)
    mechanism_supported = primary[
        primary["direction_compatible_chembl_match"] & primary["preservation_gate_pass"]
    ].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT / "level5_frozen_module_coverage.tsv", sep="\t", index=False)
    raw_scores.to_csv(OUT / "lincs_signature_raw_module_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    signature_state.to_csv(OUT / "lincs_signature_state_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    cell.to_csv(OUT / "lincs_compound_cell_state_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    compound.to_csv(OUT / "lincs_compound_state_rankings_all_times.tsv.gz", sep="\t", index=False, compression="gzip")
    top.to_csv(OUT / "lincs_top100_per_state_24h.tsv", sep="\t", index=False)
    mechanism_supported.to_csv(OUT / "lincs_mechanism_supported_candidates_24h.tsv", sep="\t", index=False)

    tier_counts = (
        primary.groupby(["state", "evidence_tier"]).size().rename("compounds").reset_index()
    )
    tier_counts.to_csv(OUT / "lincs_primary_evidence_tier_counts.tsv", sep="\t", index=False)
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS",
        "analysis_type": "state-specific breast-cell LINCS target-reversal with immune-preservation penalty",
        "whole_state_reversal_primary": False,
        "signatures_scored": int(len(raw_scores)),
        "cell_lines": int(signatures["cell_id"].nunique()),
        "compounds": int(signatures["pert_id"].nunique()),
        "states": sorted(modules["state"].unique()),
        "module_scores": int(len(module_columns)),
        "primary_time_h": PRIMARY_TIME_H,
        "primary_ranking_compounds": {str(k): int(v) for k, v in primary.groupby("state")["pert_id"].nunique().items()},
        "primary_preservation_pass": {str(k): int(v) for k, v in primary[primary["preservation_gate_pass"]].groupby("state")["pert_id"].nunique().items()},
        "primary_mechanism_supported": {str(k): int(v) for k, v in mechanism_supported.groupby("state")["pert_id"].nunique().items()},
        "preservation_margin": PRESERVATION_MARGIN,
        "preservation_gate": PRESERVATION_GATE,
        "minimum_cell_lines_for_ranking": MIN_CELLS_FOR_RANKING,
        "epithelial_interpretation": "exploratory only; requires PRISM/GDSC dependency validation",
        "mesenchymal_interpretation": "cell-line perturbation supports malignant-executable direction only; stromal/immune efficacy remains unproven",
        "next_gate": "inspect state-specific candidates and failure modes, then add PRISM/GDSC sensitivity and drug-combination complementarity",
    }
    (OUT / "state_specific_lincs_scoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
