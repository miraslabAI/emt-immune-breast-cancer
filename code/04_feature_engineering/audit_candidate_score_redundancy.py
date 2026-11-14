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

# Locked before inspecting results. No representative is selected here: this
# audit only identifies candidate redundancy groups for a later documented lock.
PAIR_REPORT_ABS_RHO = 0.80
REDUNDANCY_COMPONENT_ABS_RHO = 0.85
NEAR_SINGULAR_CONDITION_INDEX = 30.0

DEVELOPMENT_COHORTS = [
    ("survival", "METABRIC"),
    ("response", "GSE25055"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def connected_components(correlation: pd.DataFrame, threshold: float) -> list[list[str]]:
    names = correlation.columns.tolist()
    adjacency = {name: set() for name in names}
    values = correlation.to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(values[i, j]) >= threshold:
                adjacency[names[i]].add(names[j])
                adjacency[names[j]].add(names[i])
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in names:
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if len(component) > 1:
            components.append(sorted(component))
    return sorted(components, key=lambda x: (-len(x), x))


def condition_diagnostics(frame: pd.DataFrame) -> tuple[float, int, list[float]]:
    x = frame.to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=0)
    if np.any(sd == 0):
        return float("inf"), 0, []
    x = (x - x.mean(axis=0)) / sd
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    tolerance = max(x.shape) * np.finfo(float).eps * singular_values[0]
    positive = singular_values[singular_values > tolerance]
    numerical_rank = int(len(positive))
    condition_index = float(singular_values[0] / positive[-1]) if len(positive) else float("inf")
    return condition_index, numerical_rank, singular_values.tolist()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_components: list[dict] = []
    hard_failures: list[str] = []
    review_flags: list[str] = []

    for branch, cohort in DEVELOPMENT_COHORTS:
        path = SCORE_DIR / f"{cohort}_{branch}_rank_scores.tsv.gz"
        frame = pd.read_csv(path, sep="\t", index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        values = frame.to_numpy(dtype=float)
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"Duplicate identifiers in {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {path}")

        zero_variance = frame.columns[frame.std(axis=0, ddof=0) < 1e-8].tolist()
        if zero_variance:
            hard_failures.append(f"{branch}/{cohort}: zero-variance features: {zero_variance}")

        correlation = frame.corr(method="spearman")
        corr_out = OUT_DIR / f"{cohort}_{branch}_development_spearman.tsv"
        correlation.to_csv(corr_out, sep="\t", index=True)

        rows, cols = np.triu_indices(correlation.shape[0], k=1)
        pair_table = pd.DataFrame({
            "feature_1": correlation.index.to_numpy()[rows],
            "feature_2": correlation.columns.to_numpy()[cols],
            "spearman_rho": correlation.to_numpy()[rows, cols],
        })
        pair_table["abs_spearman_rho"] = pair_table["spearman_rho"].abs()
        pair_table = pair_table.loc[pair_table["abs_spearman_rho"] >= PAIR_REPORT_ABS_RHO]
        pair_table = pair_table.sort_values("abs_spearman_rho", ascending=False)
        pair_out = OUT_DIR / f"{cohort}_{branch}_high_correlation_pairs.tsv"
        pair_table.to_csv(pair_out, sep="\t", index=False)

        components = connected_components(correlation, REDUNDANCY_COMPONENT_ABS_RHO)
        for component_id, members in enumerate(components, start=1):
            all_components.append({
                "branch": branch,
                "development_cohort": cohort,
                "component_id": component_id,
                "n_features": len(members),
                "members": members,
                "representative_selected": None,
            })

        condition_index, numerical_rank, singular_values = condition_diagnostics(frame)
        if condition_index >= NEAR_SINGULAR_CONDITION_INDEX:
            review_flags.append(f"{branch}/{cohort}: condition index={condition_index:.2f}")
        if components:
            review_flags.append(f"{branch}/{cohort}: {len(components)} redundancy components")

        summaries.append({
            "branch": branch,
            "development_cohort": cohort,
            "samples": int(frame.shape[0]),
            "features": int(frame.shape[1]),
            "input_file": str(path),
            "input_sha256": sha256(path),
            "reported_pair_count_abs_rho_ge_0_80": int(len(pair_table)),
            "redundancy_component_count_abs_rho_ge_0_85": int(len(components)),
            "features_in_redundancy_components": int(sum(len(x) for x in components)),
            "numerical_rank": numerical_rank,
            "condition_index": condition_index,
            "singular_values": singular_values,
            "correlation_matrix_file": str(corr_out),
            "high_correlation_pairs_file": str(pair_out),
        })

    components_path = OUT_DIR / "candidate_redundancy_components.json"
    components_path.write_text(json.dumps(all_components, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "FAIL" if hard_failures else ("REVIEW" if review_flags else "PASS")
    summary = {
        "status": status,
        "version": "1.0",
        "analysis_type": "outcome-blind development-cohort candidate-score redundancy QC",
        "outcomes_used": False,
        "representatives_selected": False,
        "locked_thresholds": {
            "pair_report_abs_spearman_rho": PAIR_REPORT_ABS_RHO,
            "redundancy_component_abs_spearman_rho": REDUNDANCY_COMPONENT_ABS_RHO,
            "near_singular_condition_index": NEAR_SINGULAR_CONDITION_INDEX,
        },
        "cohorts": summaries,
        "hard_failures": hard_failures,
        "review_flags": review_flags,
        "redundancy_components_file": str(components_path),
        "next_gate": "joint_review_with_candidate_score_transport_qc",
    }
    summary_path = OUT_DIR / "candidate_score_redundancy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
