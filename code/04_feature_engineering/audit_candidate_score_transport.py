from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCORE_DIR = ROOT / "02_data/03_processed/analysis_ready_v1/candidate_rank_scores"
OUT_DIR = ROOT / "04_analysis/03_feature_engineering/candidate_score_qc"

# Locked before inspecting the results. These are representation-QC flags, not
# biological effect or outcome-selection thresholds.
MAX_ABS_MEAN_SHIFT_SD = 1.0
MIN_SD_RATIO = 0.50
MAX_SD_RATIO = 2.00
MIN_DEVELOPMENT_SD = 1e-8
STRONG_CORRELATION = 0.50
SIGN_FLIP_VALIDATION_MIN_ABS_RHO = 0.30
MIN_CORRELATION_STRUCTURE_CONCORDANCE = 0.60

COMPARISONS = [
    ("survival", "METABRIC", "TCGA_BRCA"),
    ("survival", "METABRIC", "SCANB_GSE96058"),
    ("response", "GSE25055", "GSE25065"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_path(cohort: str, branch: str) -> Path:
    return SCORE_DIR / f"{cohort}_{branch}_rank_scores.tsv.gz"


def load_scores(cohort: str, branch: str) -> tuple[pd.DataFrame, Path]:
    path = score_path(cohort, branch)
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    values = frame.to_numpy(dtype=float)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"Duplicate identifiers in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite score in {path}")
    return frame, path


def upper_triangle(correlation: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]]]:
    rows, cols = np.triu_indices(correlation.shape[0], k=1)
    pairs = [(correlation.index[i], correlation.columns[j]) for i, j in zip(rows, cols)]
    return correlation.to_numpy()[rows, cols], pairs


def correlation_concordance(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_tables: list[pd.DataFrame] = []
    comparison_summaries: list[dict] = []
    input_manifest: list[dict] = []
    hard_failures: list[str] = []
    review_flags: list[str] = []

    for branch, development_name, validation_name in COMPARISONS:
        development, development_path = load_scores(development_name, branch)
        validation, validation_path = load_scores(validation_name, branch)
        if development.columns.tolist() != validation.columns.tolist():
            raise ValueError(f"Feature order mismatch: {development_name} vs {validation_name}")

        for cohort, path, frame in [
            (development_name, development_path, development),
            (validation_name, validation_path, validation),
        ]:
            item = {"branch": branch, "cohort": cohort, "path": str(path),
                    "sha256": sha256(path), "samples": int(frame.shape[0]),
                    "features": int(frame.shape[1])}
            if item not in input_manifest:
                input_manifest.append(item)

        dev_mean = development.mean(axis=0)
        dev_sd = development.std(axis=0, ddof=0)
        dev_iqr = development.quantile(0.75) - development.quantile(0.25)
        val_mean = validation.mean(axis=0)
        val_sd = validation.std(axis=0, ddof=0)
        val_iqr = validation.quantile(0.75) - validation.quantile(0.25)

        table = pd.DataFrame({
            "branch": branch,
            "development_cohort": development_name,
            "validation_cohort": validation_name,
            "feature": development.columns,
            "development_mean": dev_mean.to_numpy(),
            "development_sd": dev_sd.to_numpy(),
            "development_iqr": dev_iqr.to_numpy(),
            "validation_mean": val_mean.to_numpy(),
            "validation_sd": val_sd.to_numpy(),
            "validation_iqr": val_iqr.to_numpy(),
        })
        table["mean_shift_in_development_sd"] = (
            (table["validation_mean"] - table["development_mean"]) /
            table["development_sd"].replace(0, np.nan)
        )
        table["sd_ratio"] = table["validation_sd"] / table["development_sd"].replace(0, np.nan)
        table["iqr_ratio"] = table["validation_iqr"] / table["development_iqr"].replace(0, np.nan)
        table["near_zero_development_variance"] = table["development_sd"] < MIN_DEVELOPMENT_SD
        table["mean_shift_flag"] = table["mean_shift_in_development_sd"].abs() > MAX_ABS_MEAN_SHIFT_SD
        table["sd_ratio_flag"] = ~table["sd_ratio"].between(MIN_SD_RATIO, MAX_SD_RATIO, inclusive="both")
        table["transport_flag"] = np.where(
            table[["near_zero_development_variance", "mean_shift_flag", "sd_ratio_flag"]].any(axis=1),
            "REVIEW", "PASS"
        )
        feature_tables.append(table)

        dev_corr = development.corr(method="spearman")
        val_corr = validation.corr(method="spearman")
        dev_vector, pairs = upper_triangle(dev_corr)
        val_vector, _ = upper_triangle(val_corr)
        concordance = correlation_concordance(dev_vector, val_vector)

        pair_table = pd.DataFrame({
            "feature_1": [p[0] for p in pairs],
            "feature_2": [p[1] for p in pairs],
            "development_spearman_rho": dev_vector,
            "validation_spearman_rho": val_vector,
        })
        pair_table["absolute_rho_change"] = (
            pair_table["validation_spearman_rho"] - pair_table["development_spearman_rho"]
        ).abs()
        pair_table["strong_sign_flip"] = (
            (pair_table["development_spearman_rho"].abs() >= STRONG_CORRELATION) &
            (pair_table["validation_spearman_rho"].abs() >= SIGN_FLIP_VALIDATION_MIN_ABS_RHO) &
            (np.sign(pair_table["development_spearman_rho"]) !=
             np.sign(pair_table["validation_spearman_rho"]))
        )
        pair_out = OUT_DIR / f"{branch}_{development_name}_vs_{validation_name}_pair_transport.tsv"
        pair_table.to_csv(pair_out, sep="\t", index=False)

        zero_variance = table.loc[table["near_zero_development_variance"], "feature"].tolist()
        if zero_variance:
            hard_failures.append(f"{branch}/{development_name}: zero-variance features: {zero_variance}")
        flagged = table.loc[table["transport_flag"] == "REVIEW", "feature"].tolist()
        sign_flips = pair_table.loc[pair_table["strong_sign_flip"]]
        if flagged:
            review_flags.append(f"{branch}/{validation_name}: {len(flagged)} feature transport flags")
        if np.isfinite(concordance) and concordance < MIN_CORRELATION_STRUCTURE_CONCORDANCE:
            review_flags.append(f"{branch}/{validation_name}: correlation concordance={concordance:.3f}")
        if len(sign_flips):
            review_flags.append(f"{branch}/{validation_name}: {len(sign_flips)} strong sign flips")

        comparison_summaries.append({
            "branch": branch,
            "development_cohort": development_name,
            "validation_cohort": validation_name,
            "development_samples": int(development.shape[0]),
            "validation_samples": int(validation.shape[0]),
            "features": int(development.shape[1]),
            "feature_transport_review_count": int(len(flagged)),
            "correlation_structure_concordance": concordance,
            "strong_sign_flip_count": int(len(sign_flips)),
            "pair_transport_file": str(pair_out),
        })

    feature_qc = pd.concat(feature_tables, ignore_index=True)
    feature_out = OUT_DIR / "candidate_feature_transport_qc.tsv"
    feature_qc.to_csv(feature_out, sep="\t", index=False)

    status = "FAIL" if hard_failures else ("REVIEW" if review_flags else "PASS")
    summary = {
        "status": status,
        "version": "1.0",
        "analysis_type": "outcome-blind candidate-score transport QC",
        "outcomes_used": False,
        "locked_thresholds": {
            "max_abs_mean_shift_in_development_sd": MAX_ABS_MEAN_SHIFT_SD,
            "permitted_sd_ratio": [MIN_SD_RATIO, MAX_SD_RATIO],
            "minimum_development_sd": MIN_DEVELOPMENT_SD,
            "strong_development_correlation_abs_rho": STRONG_CORRELATION,
            "sign_flip_validation_min_abs_rho": SIGN_FLIP_VALIDATION_MIN_ABS_RHO,
            "minimum_correlation_structure_concordance": MIN_CORRELATION_STRUCTURE_CONCORDANCE,
        },
        "inputs": input_manifest,
        "comparisons": comparison_summaries,
        "hard_failures": hard_failures,
        "review_flags": review_flags,
        "feature_qc_file": str(feature_out),
        "next_gate": "joint_review_with_candidate_score_redundancy_qc",
    }
    summary_path = OUT_DIR / "candidate_score_transport_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
