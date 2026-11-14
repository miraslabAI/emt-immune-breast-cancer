#!/usr/bin/env python
"""Score frozen, outcome-blind and interpretable state-specific therapy hypotheses."""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
IN_DIR = ROOT / "04_analysis/11_drug_repositioning/20_therapy_prioritization_feature_lock"
OUT = ROOT / "04_analysis/11_drug_repositioning/21_state_specific_therapy_scores"
COHORTS = ["TCGA_BRCA", "METABRIC", "SCANB_GSE96058"]
EPS = 1e-6

# Each top-level component is one biological AND-gate. Correlated readouts within a
# component are averaged first, preventing large pathway families from dominating.
FORMULAS = {
    "hybrid_panERBB_AKT": {
        "therapy_hypothesis": "pan-ErbB plus AKT inhibition",
        "components": {
            "hybrid_state": ["em_coactivation"],
            "erbb_dependency": ["erbb2_ptk6"],
            "akt_pathway": ["pi3k_akt_mtor"],
        },
    },
    "hybrid_panERBB_mTORC12": {
        "therapy_hypothesis": "pan-ErbB plus dual mTORC1/2 inhibition",
        "components": {
            "hybrid_state": ["em_coactivation"],
            "erbb_dependency": ["erbb2_ptk6"],
            "mtor_pathway": ["mtorc1"],
        },
    },
    "mesenchymal_antiangiogenic_IO": {
        "therapy_hypothesis": "low-dose FGFR/VEGFR remodeling plus PD-1/PD-L1 blockade",
        "components": {
            "mesenchymal_state": ["mesenchymal_core", "emt_hallmark"],
            "vascular_stromal_barrier": ["angiogenesis", "hypoxia", "tgfb", "stromal_z"],
            "immune_visibility": ["antigen_presentation", "priming_recruitment", "infiltration", "ifng_response"],
            "incomplete_completion": ["immune_exclusion_pressure", "suppressive_recruitment"],
        },
    },
}


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    """Deterministic within-cohort percentile in (0, 1), average-rank for ties."""
    return frame[column].rank(method="average", pct=True).clip(EPS, 1.0)


def weighted_geomean(component_frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    names = list(component_frame.columns)
    w = np.asarray([weights[name] for name in names], dtype=float)
    x = component_frame[names].to_numpy(dtype=float)
    return np.exp((np.log(np.clip(x, EPS, 1.0)) * w).sum(axis=1) / w.sum())


def build_components(frame: pd.DataFrame, specification: dict) -> pd.DataFrame:
    result = {}
    for component, features in specification["components"].items():
        ranks = np.column_stack([percentile(frame, feature) for feature in features])
        result[component] = ranks.mean(axis=1)
    return pd.DataFrame(result, index=frame.index)


def top_decile(values: np.ndarray) -> np.ndarray:
    cutoff = np.quantile(values, 0.90)
    return values >= cutoff


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    formula_rows = []
    for formula, specification in FORMULAS.items():
        for component, features in specification["components"].items():
            formula_rows.append({
                "formula": formula,
                "therapy_hypothesis": specification["therapy_hypothesis"],
                "component": component,
                "features_averaged_within_component": ";".join(features),
                "baseline_component_weight": 1.0,
                "combination_rule": "weighted geometric mean across components (AND-gate)",
            })
    pd.DataFrame(formula_rows).to_csv(OUT / "frozen_therapy_hypothesis_formulas.tsv", sep="\t", index=False)

    score_frames = []
    sensitivity_rows = []
    top_state_rows = []
    cohort_rows = []
    selection_rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(IN_DIR / f"{cohort}_therapy_prioritization_features.tsv.gz", sep="\t", low_memory=False)
        cohort_scores = frame[["sample_id", "cohort", "immune_state", "pam50"]].copy()
        for formula, specification in FORMULAS.items():
            components = build_components(frame, specification)
            baseline_weights = {name: 1.0 for name in components.columns}
            baseline = weighted_geomean(components, baseline_weights)
            cohort_scores[f"score__{formula}"] = baseline
            base_top = top_decile(baseline)
            cohort_scores[f"top10__{formula}"] = base_top

            # Full prespecified 0.5/1/2 grid: tests whether rankings depend on an
            # arbitrary component emphasis. No outcomes enter this analysis.
            alternatives = []
            top_counts = np.zeros(len(frame), dtype=int)
            grid = list(itertools.product([0.5, 1.0, 2.0], repeat=components.shape[1]))
            for values in grid:
                weights = dict(zip(components.columns, values))
                alternative = weighted_geomean(components, weights)
                alt_top = top_decile(alternative)
                top_counts += alt_top.astype(int)
                rho = float(spearmanr(baseline, alternative).statistic)
                union = np.logical_or(base_top, alt_top).sum()
                jaccard = float(np.logical_and(base_top, alt_top).sum() / union) if union else 1.0
                alternatives.append((rho, jaccard))
                sensitivity_rows.append({
                    "cohort": cohort,
                    "formula": formula,
                    "weights": ";".join(f"{k}={v:g}" for k, v in weights.items()),
                    "spearman_vs_equal_weight": rho,
                    "top10_jaccard_vs_equal_weight": jaccard,
                })
            cohort_scores[f"selection_frequency__{formula}"] = top_counts / len(grid)
            arr = np.asarray(alternatives)
            cohort_rows.append({
                "cohort": cohort,
                "formula": formula,
                "samples": len(frame),
                "weight_grid_size": len(grid),
                "min_spearman": arr[:, 0].min(),
                "median_spearman": np.median(arr[:, 0]),
                "min_top10_jaccard": arr[:, 1].min(),
                "median_top10_jaccard": np.median(arr[:, 1]),
                "stable_core_n_frequency_ge_0_80": int((top_counts / len(grid) >= 0.80).sum()),
            })
            for state, count in frame.loc[base_top, "immune_state"].value_counts().items():
                top_state_rows.append({
                    "cohort": cohort, "formula": formula, "immune_state": state,
                    "top10_n": int(count), "top10_fraction": float(count / base_top.sum()),
                    "cohort_fraction": float((frame["immune_state"] == state).mean()),
                    "enrichment_ratio": float((count / base_top.sum()) / (frame["immune_state"] == state).mean()),
                })
            for sample_id, frequency in zip(frame["sample_id"], top_counts / len(grid)):
                if frequency >= 0.80:
                    selection_rows.append({"cohort": cohort, "sample_id": sample_id, "formula": formula,
                                           "selection_frequency": frequency})
        score_frames.append(cohort_scores)

    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(OUT / "sample_level_therapy_hypothesis_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "weight_sensitivity_grid.tsv.gz", sep="\t", index=False, compression="gzip")
    cohort_frame = pd.DataFrame(cohort_rows)
    cohort_frame.to_csv(OUT / "therapy_score_stability_summary.tsv", sep="\t", index=False)
    pd.DataFrame(top_state_rows).to_csv(OUT / "top_decile_immune_state_enrichment.tsv", sep="\t", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "stable_core_samples.tsv.gz", sep="\t", index=False, compression="gzip")

    gates_passed = bool(
        np.isfinite(scores.filter(like="score__").to_numpy()).all()
        and (cohort_frame["min_spearman"] >= 0.70).all()
    )
    summary = {
        "status": "PASS" if gates_passed else "REVIEW",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind continuous therapy-hypothesis prioritization with weight sensitivity",
        "samples": int(len(scores)),
        "cohorts": COHORTS,
        "formulas": {key: value["therapy_hypothesis"] for key, value in FORMULAS.items()},
        "logic": "within-cohort percentile ranks; correlated features averaged within biological domains; geometric mean across required domains",
        "weight_sensitivity_grid": [0.5, 1.0, 2.0],
        "minimum_required_spearman": 0.70,
        "all_rank_stability_gates_passed": gates_passed,
        "outcomes_used": False,
        "weights_fitted": False,
        "treatment_response_claimed": False,
        "interpretation": "hypothesis eligibility/ranking, not a treatment-response predictor",
        "next_gate": "inspect cross-cohort stability and immune-state enrichment, then lock biomarker panels and prospective validation design",
    }
    (OUT / "therapy_score_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
