#!/usr/bin/env python
"""Analyze completed Hybrid dose-matrix plates under the frozen prospective contract."""
from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK = ROOT / "04_analysis/11_drug_repositioning/28_hybrid_plate_analysis_lock"
CONTRACT = ROOT / "04_analysis/11_drug_repositioning/29_hybrid_result_ingestion_contract"
DEFAULT_OUT = ROOT / "04_analysis/11_drug_repositioning/30_hybrid_combination_results"


def holm(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * float(pvalues.iloc[index]))
        running = max(running, value)
        adjusted[index] = running
    return pd.Series(adjusted, index=pvalues.index)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iterations: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).tolist())


def validate_and_load(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = input_dir / "completed_raw_plate_results.tsv"
    metadata_path = input_dir / "completed_plate_metadata.tsv"
    if not raw_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Both completed_raw_plate_results.tsv and completed_plate_metadata.tsv are required")
    raw = pd.read_csv(raw_path, sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t")
    template = pd.read_csv(CONTRACT / "raw_plate_results_template.tsv", sep="\t")
    expected = template[["plate_id", "well"]].sort_values(["plate_id", "well"]).reset_index(drop=True)
    observed = raw[["plate_id", "well"]].sort_values(["plate_id", "well"]).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError("Completed raw results do not exactly match the locked plate/well keys")
    if raw.duplicated(["plate_id", "well"]).any():
        raise ValueError("Duplicate plate/well keys detected")
    raw["raw_signal"] = pd.to_numeric(raw["raw_signal"], errors="coerce")
    measured = raw["read_status"].eq("measured")
    if raw.loc[measured, "raw_signal"].isna().any() or (raw.loc[measured, "raw_signal"] < 0).any():
        raise ValueError("Measured wells require finite, non-negative raw_signal values")
    excluded_without_reason = raw["read_status"].eq("excluded") & raw["exclusion_reason"].fillna("").eq("")
    if excluded_without_reason.any():
        raise ValueError("Every excluded well requires an exclusion_reason")
    expected_plates = set(expected["plate_id"])
    if set(metadata["plate_id"]) != expected_plates or metadata["plate_id"].duplicated().any():
        raise ValueError("Plate metadata must contain every locked plate exactly once")
    return raw, metadata


def plate_analysis(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc_rows, normalized_rows, score_rows = [], [], []
    for plate_id, frame in raw.groupby("plate_id", sort=False):
        used = frame[frame["read_status"].eq("measured")].copy()
        blank = used[used["well_role"].isin(["media_blank", "no_cell_blank"])]["raw_signal"]
        vehicle = used[used["well_role"].eq("vehicle_control")]["raw_signal"]
        kill = used[used["well_role"].eq("positive_kill_control")]["raw_signal"]
        complete_controls = min(len(blank), len(vehicle), len(kill)) >= 3
        blank_median = float(blank.median()) if complete_controls else np.nan
        vehicle_bs = vehicle - blank_median
        kill_bs = kill - blank_median
        vehicle_cv = float(vehicle_bs.std(ddof=1) / vehicle_bs.mean()) if complete_controls and vehicle_bs.mean() > 0 else np.inf
        denominator = abs(vehicle_bs.mean() - kill_bs.mean()) if complete_controls else np.nan
        z_prime = 1 - 3 * (vehicle_bs.std(ddof=1) + kill_bs.std(ddof=1)) / denominator if denominator and denominator > 0 else -np.inf
        qc_pass = bool(complete_controls and vehicle_cv <= 0.20 and z_prime >= 0.50)
        identity = frame.iloc[0]
        qc_rows.append({
            "plate_id": plate_id, "cell_line": identity.cell_line, "pair_id": identity.pair_id,
            "biological_repeat": identity.biological_repeat, "blank_median": blank_median,
            "vehicle_cv": vehicle_cv, "z_prime": z_prime, "plate_qc_pass": qc_pass,
        })
        if not qc_pass:
            continue
        used["blank_subtracted_signal"] = used["raw_signal"] - blank_median
        vehicle_median = float(vehicle_bs.median())
        used["normalized_viability"] = used["blank_subtracted_signal"] / vehicle_median
        used["normalized_inhibition"] = 1.0 - used["normalized_viability"]
        normalized_rows.append(used)

        matrix = used[used["well_role"].eq("dose_matrix")].copy()
        grouped = matrix.groupby(
            ["agent_A_IC50_multiple", "agent_B_IC50_multiple"], as_index=False
        )["normalized_viability"].mean()
        lookup = grouped.set_index(["agent_A_IC50_multiple", "agent_B_IC50_multiple"])["normalized_viability"]
        cell_scores = []
        for (dose_a, dose_b), observed_viability in lookup.items():
            if dose_a <= 0 or dose_b <= 0:
                continue
            viability_a = float(lookup.loc[(dose_a, 0.0)])
            viability_b = float(lookup.loc[(0.0, dose_b)])
            bliss = 100.0 * (viability_a * viability_b - observed_viability)
            hsa = 100.0 * (min(viability_a, viability_b) - observed_viability)
            cell_scores.append((bliss, hsa))
        if len(cell_scores) != 25:
            raise ValueError(f"{plate_id} does not contain the complete locked 5x5 non-zero grid")
        cell_scores = np.asarray(cell_scores)
        score_rows.append({
            "plate_id": plate_id, "cell_line": identity.cell_line, "pair_id": identity.pair_id,
            "biological_repeat": identity.biological_repeat,
            "mean_bliss_excess": float(cell_scores[:, 0].mean()),
            "mean_hsa_excess": float(cell_scores[:, 1].mean()),
        })
    normalized = pd.concat(normalized_rows, ignore_index=True) if normalized_rows else pd.DataFrame()
    return pd.DataFrame(qc_rows), normalized, pd.DataFrame(score_rows)


def summarize(scores: pd.DataFrame, output_dir: Path, seed: int) -> None:
    models = pd.read_csv(LOCK / "primary_backup_model_lock.tsv", sep="\t")
    roles = models[["cell_line", "role", "biological_role"]].drop_duplicates("cell_line")
    scores = scores.merge(roles, on="cell_line", how="left", validate="many_to_one")
    rng = np.random.default_rng(seed)
    model_rows = []
    for (cell_line, pair_id), frame in scores.groupby(["cell_line", "pair_id"]):
        values = frame.sort_values("biological_repeat")["mean_bliss_excess"].to_numpy()
        if len(values) != 3:
            raise ValueError(f"{cell_line}/{pair_id} requires all three biological repeats after QC")
        low, high = bootstrap_ci(values, rng)
        role = frame["role"].iloc[0]
        model_rows.append({
            "cell_line": cell_line, "pair_id": pair_id, "role": role,
            "median_bliss": float(np.median(values)), "bootstrap_ci_low": low,
            "bootstrap_ci_high": high, "all_repeats_positive": bool((values > 0).all()),
            "model_positive": bool(np.median(values) > 10 and low > 0 and (values > 0).all()),
        })
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(output_dir / "model_level_primary_synergy.tsv", sep="\t", index=False)

    pair_rows = []
    for pair_id, frame in model_summary.groupby("pair_id"):
        core = frame[frame["role"].eq("core_primary")]
        controls = frame[frame["role"].eq("state_anchor_control")]
        positive_core = core[core["model_positive"]]
        has_her2 = positive_core["cell_line"].isin(["HCC1954", "MDAMB361"]).any()
        has_tnbc = positive_core["cell_line"].isin(["BT20", "HCC70"]).any()
        pair_advance = len(positive_core) >= 3 and has_her2 and has_tnbc
        core_values = core["median_bliss"].to_numpy()
        control_values = controls["median_bliss"].to_numpy()
        contrast = float(core_values.mean() - control_values.mean())
        draws = []
        for _ in range(10000):
            draws.append(
                rng.choice(core_values, len(core_values), replace=True).mean()
                - rng.choice(control_values, len(control_values), replace=True).mean()
            )
        try:
            pvalue = float(wilcoxon(core_values, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = 1.0
        pair_rows.append({
            "pair_id": pair_id, "positive_core_models": len(positive_core),
            "includes_HER2_positive_model": bool(has_her2), "includes_TNBC_model": bool(has_tnbc),
            "pair_advance_gate": bool(pair_advance), "core_minus_control_bliss": contrast,
            "contrast_ci_low": float(np.quantile(draws, 0.025)),
            "contrast_ci_high": float(np.quantile(draws, 0.975)),
            "one_sided_wilcoxon_p": pvalue,
        })
    pair_summary = pd.DataFrame(pair_rows)
    pair_summary["holm_adjusted_p"] = holm(pair_summary["one_sided_wilcoxon_p"])
    pair_summary.to_csv(output_dir / "pair_level_primary_decisions.tsv", sep="\t", index=False)

    decision = {
        "status": "PASS" if pair_summary["pair_advance_gate"].any() else "NO_PAIR_ADVANCED",
        "primary_metric": "mean Bliss excess over locked 5x5 non-zero grid",
        "pairs_advanced": pair_summary.loc[pair_summary["pair_advance_gate"], "pair_id"].tolist(),
        "zip_loewe_status": "supportive SynergyFinder import required; not substituted for primary Bliss",
        "clinical_recommendation": False,
        "in_vitro_combination_support_established": bool(pair_summary["pair_advance_gate"].any()),
        "combination_efficacy_established": False,
        "treatment_predictive_biomarker_established": False,
    }
    (output_dir / "hybrid_combination_decision_summary.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-seed", type=int, default=20261101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, metadata = validate_and_load(args.input_dir)
    qc, normalized, scores = plate_analysis(raw)
    qc.to_csv(args.output_dir / "plate_qc.tsv", sep="\t", index=False)
    normalized.to_csv(args.output_dir / "normalized_well_results.tsv", sep="\t", index=False)
    scores.to_csv(args.output_dir / "plate_level_synergy.tsv", sep="\t", index=False)
    metadata.to_csv(args.output_dir / "ingested_plate_metadata.tsv", sep="\t", index=False)
    if not normalized.empty:
        matrix = normalized[normalized["well_role"].eq("dose_matrix")].copy()
        synergyfinder = matrix.groupby(
            ["plate_id", "agent_A", "agent_B", "agent_A_IC50_multiple", "agent_B_IC50_multiple"],
            as_index=False,
        )["normalized_inhibition"].mean()
        synergyfinder = synergyfinder.rename(columns={
            "plate_id": "block_id", "agent_A": "drug_row", "agent_B": "drug_col",
            "agent_A_IC50_multiple": "conc_r", "agent_B_IC50_multiple": "conc_c",
            "normalized_inhibition": "response_fraction",
        })
        synergyfinder["response"] = 100.0 * synergyfinder.pop("response_fraction")
        synergyfinder["conc_r_unit"] = "IC50_multiple"
        synergyfinder["conc_c_unit"] = "IC50_multiple"
        synergyfinder.to_csv(args.output_dir / "synergyfinder_long_format.tsv", sep="\t", index=False)
    summarize(scores, args.output_dir, args.bootstrap_seed)


if __name__ == "__main__":
    main()
