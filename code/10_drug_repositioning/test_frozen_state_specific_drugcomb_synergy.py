#!/usr/bin/env python
"""Test the frozen DrugComb universe for EMT-state-selective combination synergy."""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
LOCK = BASE / "16_state_specific_drugcomb_lock/frozen_state_specific_drugcomb_pair_universe.tsv"
BLOCKS = BASE / "16_state_specific_drugcomb_lock/frozen_state_specific_drugcomb_blocks.tsv.gz"
LOCK_SUMMARY = BASE / "16_state_specific_drugcomb_lock/state_specific_drugcomb_lock_summary.json"
OUT = BASE / "17_state_specific_drugcomb_synergy"
SYNERGY = ["synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
ENDPOINTS = ["consensus_synergy", *SYNERGY]


def bh(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    x = p[good]
    order = np.argsort(x)
    adjusted = x[order] * len(x) / np.arange(1, len(x) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[good] = restored
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()) / (len(x) * len(y)))


def one_sided_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    good = x.notna() & y.notna()
    if good.sum() < 5 or x[good].nunique() < 3 or y[good].nunique() < 3:
        return np.nan, np.nan
    rho, p_two = spearmanr(x[good], y[good])
    p_one = p_two / 2 if rho >= 0 else 1 - p_two / 2
    return float(rho), float(p_one)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_summary = json.loads(LOCK_SUMMARY.read_text(encoding="utf-8"))
    if lock_summary.get("outcomes_used_for_selection") is not False:
        raise ValueError("Frozen universe was not outcome-blind")
    lock = pd.read_csv(LOCK, sep="\t", low_memory=False)
    blocks = pd.read_csv(BLOCKS, sep="\t", low_memory=False)
    for column in [*SYNERGY, "css_ri", "emt_position"]:
        blocks[column] = pd.to_numeric(blocks[column], errors="coerce")

    # Median across the four standard models is the pre-specified primary
    # endpoint; individual models remain visible as secondary endpoints.
    blocks["consensus_synergy"] = blocks[SYNERGY].median(axis=1, skipna=True)
    # Biological replicates/blocks from the same model are not independent.
    # Collapse them before inference so each cell line contributes once.
    model_values = (
        blocks.groupby(
            ["locked_pair_id", "ModelID", "emt_state", "is_target_state"],
            as_index=False,
            dropna=False,
        )
        .agg(
            epithelial_core=("epithelial_core", "first"),
            mesenchymal_core=("mesenchymal_core", "first"),
            emt_position=("emt_position", "first"),
            source_blocks=("block_id", "nunique"),
            consensus_synergy=("consensus_synergy", "median"),
            synergy_zip=("synergy_zip", "median"),
            synergy_loewe=("synergy_loewe", "median"),
            synergy_hsa=("synergy_hsa", "median"),
            synergy_bliss=("synergy_bliss", "median"),
            css_ri=("css_ri", "median"),
        )
    )
    model_values.to_csv(OUT / "frozen_drugcomb_model_level_synergy.tsv", sep="\t", index=False)
    tests = []
    for pair_id, group in model_values.groupby("locked_pair_id", sort=True):
        meta = lock[lock["locked_pair_id"].eq(pair_id)].iloc[0]
        for endpoint in ENDPOINTS:
            target = group.loc[group["is_target_state"].astype(str).str.casefold().eq("true"), endpoint].dropna().to_numpy(float)
            other = group.loc[~group["is_target_state"].astype(str).str.casefold().eq("true"), endpoint].dropna().to_numpy(float)
            if len(target) < 3 or len(other) < 3:
                continue
            u = mannwhitneyu(target, other, alternative="greater", method="auto")
            rho, rho_p = one_sided_spearman(group["emt_position"], group[endpoint])
            # For hybrid, proximity to the hybrid state is represented by the
            # frozen binary contrast; EMT-position correlation is not directional.
            if meta["target_state"] == "hybrid-peak":
                rho, rho_p = np.nan, np.nan
            tests.append({
                "locked_pair_id": pair_id,
                "analysis_family": meta["analysis_family"],
                "target_state": meta["target_state"],
                "study_name": meta["study_name"],
                "anchor": meta["anchor"],
                "partner": meta["partner"],
                "prespecified_partner": meta["prespecified_partner"],
                "low_power_minimum_coverage": meta["low_power_minimum_coverage"],
                "endpoint": endpoint,
                "n_target": len(target),
                "n_other": len(other),
                "target_median": float(np.median(target)),
                "other_median": float(np.median(other)),
                "target_minus_other_median": float(np.median(target) - np.median(other)),
                "cliffs_delta": cliffs_delta(target, other),
                "mannwhitney_one_sided_p": float(u.pvalue),
                "emt_position_spearman_rho": rho,
                "emt_position_one_sided_p": rho_p,
            })

    tests = pd.DataFrame(tests)
    primary = tests["endpoint"].eq("consensus_synergy")
    tests["primary_fdr_bh_global"] = np.nan
    tests.loc[primary, "primary_fdr_bh_global"] = bh(tests.loc[primary, "mannwhitney_one_sided_p"])
    tests["endpoint_fdr_bh_within_family"] = (
        tests.groupby(["analysis_family", "endpoint"], group_keys=False)["mannwhitney_one_sided_p"]
        .transform(lambda x: bh(x))
    )
    tests.to_csv(OUT / "frozen_drugcomb_state_synergy_tests.tsv", sep="\t", index=False)

    rows = []
    for pair_id, group in tests.groupby("locked_pair_id", sort=True):
        primary_row = group[group["endpoint"].eq("consensus_synergy")].iloc[0]
        secondary = group[group["endpoint"].isin(SYNERGY)]
        positive_metrics = int((secondary["cliffs_delta"] > 0).sum())
        nominal_metrics = int((secondary["mannwhitney_one_sided_p"] < 0.05).sum())
        q = float(primary_row["primary_fdr_bh_global"])
        delta = float(primary_row["cliffs_delta"])
        if q < 0.10 and delta >= 0.33:
            evidence = "global_fdr_state_selective_synergy"
        elif primary_row["mannwhitney_one_sided_p"] < 0.05 and positive_metrics >= 3:
            evidence = "nominal_consensus_multi_metric_direction"
        elif positive_metrics >= 3 and delta > 0:
            evidence = "descriptive_multi_metric_direction"
        elif delta > 0:
            evidence = "weak_or_metric_discordant_direction"
        else:
            evidence = "no_state_selective_synergy_support"
        rows.append({
            "locked_pair_id": pair_id,
            "analysis_family": primary_row["analysis_family"],
            "target_state": primary_row["target_state"],
            "study_name": primary_row["study_name"],
            "anchor": primary_row["anchor"],
            "partner": primary_row["partner"],
            "prespecified_partner": primary_row["prespecified_partner"],
            "n_target": int(primary_row["n_target"]),
            "n_other": int(primary_row["n_other"]),
            "consensus_target_minus_other": primary_row["target_minus_other_median"],
            "consensus_cliffs_delta": delta,
            "consensus_p": primary_row["mannwhitney_one_sided_p"],
            "consensus_global_fdr": q,
            "positive_synergy_metrics": positive_metrics,
            "nominal_synergy_metrics": nominal_metrics,
            "evidence_class": evidence,
            "interpretation_constraint": "observational cell-line synergy; immune partners not evaluated; low-power rows require independent validation",
        })
    synthesis = pd.DataFrame(rows).sort_values(
        ["evidence_class", "consensus_global_fdr", "consensus_cliffs_delta"],
        ascending=[True, True, False],
    )
    synthesis.to_csv(OUT / "state_specific_drugcomb_synergy_synthesis.tsv", sep="\t", index=False)
    synthesis.groupby(["analysis_family", "evidence_class"]).size().rename("pairs").reset_index().to_csv(
        OUT / "drugcomb_synergy_evidence_counts.tsv", sep="\t", index=False
    )
    synthesis.groupby("analysis_family", group_keys=False).head(50).to_csv(
        OUT / "drugcomb_top50_per_family.tsv", sep="\t", index=False
    )

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "frozen state-specific DrugComb synergy testing",
        "primary_endpoint": "within-block median of ZIP, Loewe, HSA, and Bliss synergy",
        "primary_contrast": "target EMT state greater than other determinate EMT states",
        "locked_pairs_tested": int(synthesis["locked_pair_id"].nunique()),
        "primary_global_fdr_lt_0.10": int((synthesis["consensus_global_fdr"] < 0.10).sum()),
        "nominal_primary_p_lt_0.05": int((synthesis["consensus_p"] < 0.05).sum()),
        "prespecified_celecoxib_pairs_tested": int(synthesis["prespecified_partner"].astype(str).str.casefold().eq("true").sum()),
        "immune_synergy_evaluated": False,
        "next_gate": "integrate only corrected or independently replicated pairs; treat all others as computational hypotheses",
    }
    (OUT / "state_specific_drugcomb_synergy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
