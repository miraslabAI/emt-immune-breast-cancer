#!/usr/bin/env python
"""Robustness analysis for the frozen Hybrid DrugComb partner universe.

This analysis is explicitly secondary: it tests subtype confounding, restriction
to basal-A models, leave-one-Hybrid-model-out stability, and replication of a
partner across different ERBB anchors.  It cannot convert a discovery-set signal
into independent validation.
"""
from __future__ import annotations

import os

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "04_analysis/11_drug_repositioning"
MODEL_VALUES = BASE / "17_state_specific_drugcomb_synergy/frozen_drugcomb_model_level_synergy.tsv"
LOCK = BASE / "16_state_specific_drugcomb_lock/frozen_state_specific_drugcomb_pair_universe.tsv"
STATES = BASE / "10_pharmacogenomic_validation_lock/frozen_ccle_breast_emt_states.tsv"
SYNTHESIS = BASE / "17_state_specific_drugcomb_synergy/state_specific_drugcomb_synergy_synthesis.tsv"
OUT = BASE / "18_hybrid_drugcomb_robustness"
FAMILY = "hybrid_erbb_partner_discovery"


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


def coarse_subtype(value: object) -> str:
    text = str(value).casefold()
    if "basal_a" in text:
        return "basal_A"
    if "basal_b" in text:
        return "basal_B"
    if "her2" in text:
        return "HER2_nonbasal"
    if "er" in text or "luminal" in text:
        return "luminal_ER"
    return "other"


def one_sided_from_two(beta: float, p_two: float) -> float:
    return float(p_two / 2 if beta >= 0 else 1 - p_two / 2)


def adjusted_test(group: pd.DataFrame, cluster: bool = False) -> dict[str, object]:
    x = pd.DataFrame({"hybrid_state": group["hybrid_state"].astype(float)}, index=group.index)
    subtype = pd.get_dummies(group["coarse_subtype"], prefix="subtype", drop_first=True, dtype=float)
    x = pd.concat([x, subtype], axis=1)
    if "anchor_norm" in group and group["anchor_norm"].nunique() > 1:
        anchor = pd.get_dummies(group["anchor_norm"], prefix="anchor", drop_first=True, dtype=float)
        x = pd.concat([x, anchor], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = pd.to_numeric(group["consensus_synergy"], errors="coerce")
    good = y.notna() & x.notna().all(axis=1)
    x, y = x.loc[good], y.loc[good]
    if len(y) <= x.shape[1] + 2 or x["hybrid_state"].nunique() < 2:
        return {"beta": np.nan, "se": np.nan, "p_one": np.nan, "condition_number": np.nan, "estimable": False}
    condition = float(np.linalg.cond(x.to_numpy(float)))
    if not np.isfinite(condition) or condition >= 1e8:
        return {"beta": np.nan, "se": np.nan, "p_one": np.nan, "condition_number": condition, "estimable": False}
    model = sm.OLS(y, x).fit()
    if cluster and group.loc[good, "ModelID"].nunique() >= 5:
        robust = model.get_robustcov_results(cov_type="cluster", groups=group.loc[good, "ModelID"])
        names = list(x.columns)
        index = names.index("hybrid_state")
        beta, se, p_two = robust.params[index], robust.bse[index], robust.pvalues[index]
    else:
        robust = model.get_robustcov_results(cov_type="HC3")
        names = list(x.columns)
        index = names.index("hybrid_state")
        beta, se, p_two = robust.params[index], robust.bse[index], robust.pvalues[index]
    return {
        "beta": float(beta),
        "se": float(se),
        "p_one": one_sided_from_two(float(beta), float(p_two)),
        "condition_number": condition,
        "estimable": True,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()) / (len(x) * len(y)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    values = pd.read_csv(MODEL_VALUES, sep="\t", low_memory=False)
    lock = pd.read_csv(LOCK, sep="\t", low_memory=False)
    states = pd.read_csv(STATES, sep="\t", low_memory=False)
    discovery = pd.read_csv(SYNTHESIS, sep="\t", low_memory=False)

    lock = lock[lock["analysis_family"].eq(FAMILY)].copy()
    values = values[values["locked_pair_id"].isin(lock["locked_pair_id"])].copy()
    annotations = states[["ModelID", "ModelSubtypeFeatures"]].drop_duplicates("ModelID")
    values = values.merge(annotations, on="ModelID", how="left", validate="many_to_one")
    if values["ModelSubtypeFeatures"].isna().any():
        raise ValueError("Missing subtype annotation for frozen DrugComb models")
    values["coarse_subtype"] = values["ModelSubtypeFeatures"].map(coarse_subtype)
    values["hybrid_state"] = values["emt_state"].eq("hybrid-peak")
    values["consensus_synergy"] = pd.to_numeric(values["consensus_synergy"], errors="coerce")

    pair_rows = []
    for pair_id, group in values.groupby("locked_pair_id", sort=True):
        meta = lock[lock["locked_pair_id"].eq(pair_id)].iloc[0]
        adjusted = adjusted_test(group)
        basal = group[group["coarse_subtype"].eq("basal_A")]
        bx = basal.loc[basal["hybrid_state"], "consensus_synergy"].dropna().to_numpy(float)
        by = basal.loc[~basal["hybrid_state"], "consensus_synergy"].dropna().to_numpy(float)
        if len(bx) >= 3 and len(by) >= 3:
            basal_delta = cliffs_delta(bx, by)
            basal_p = float(mannwhitneyu(bx, by, alternative="greater", method="auto").pvalue)
        else:
            basal_delta, basal_p = np.nan, np.nan

        target_ids = group.loc[group["hybrid_state"], "ModelID"].unique()
        loo_differences = []
        loo_deltas = []
        for model_id in target_ids:
            left = group[group["ModelID"].ne(model_id)]
            x = left.loc[left["hybrid_state"], "consensus_synergy"].dropna().to_numpy(float)
            y = left.loc[~left["hybrid_state"], "consensus_synergy"].dropna().to_numpy(float)
            if len(x) and len(y):
                loo_differences.append(float(np.median(x) - np.median(y)))
                loo_deltas.append(cliffs_delta(x, y))

        disc = discovery[discovery["locked_pair_id"].eq(pair_id)].iloc[0]
        pair_rows.append({
            "locked_pair_id": pair_id,
            "anchor": meta["anchor"],
            "partner": meta["partner"],
            "n_models": int(group["ModelID"].nunique()),
            "n_hybrid": int(group.loc[group["hybrid_state"], "ModelID"].nunique()),
            "discovery_consensus_p": float(disc["consensus_p"]),
            "discovery_global_fdr": float(disc["consensus_global_fdr"]),
            "discovery_cliffs_delta": float(disc["consensus_cliffs_delta"]),
            "subtype_adjusted_beta": adjusted["beta"],
            "subtype_adjusted_se_hc3": adjusted["se"],
            "subtype_adjusted_one_sided_p": adjusted["p_one"],
            "design_condition_number": adjusted["condition_number"],
            "subtype_adjusted_estimable": adjusted["estimable"],
            "basal_A_n_hybrid": len(bx),
            "basal_A_n_other": len(by),
            "basal_A_cliffs_delta": basal_delta,
            "basal_A_one_sided_p": basal_p,
            "loo_hybrid_models": len(loo_differences),
            "loo_positive_median_fraction": float(np.mean(np.array(loo_differences) > 0)) if loo_differences else np.nan,
            "loo_positive_delta_fraction": float(np.mean(np.array(loo_deltas) > 0)) if loo_deltas else np.nan,
            "loo_min_median_difference": min(loo_differences) if loo_differences else np.nan,
            "loo_min_cliffs_delta": min(loo_deltas) if loo_deltas else np.nan,
        })

    pair_results = pd.DataFrame(pair_rows)
    pair_results["subtype_adjusted_fdr_bh"] = bh(pair_results["subtype_adjusted_one_sided_p"])
    pair_results["basal_A_fdr_bh"] = bh(pair_results["basal_A_one_sided_p"])
    pair_results["robustness_class"] = "not_robust"
    stable = (
        pair_results["subtype_adjusted_estimable"].eq(True)
        &
        pair_results["subtype_adjusted_beta"].gt(0)
        & pair_results["loo_positive_delta_fraction"].eq(1.0)
    )
    pair_results.loc[stable, "robustness_class"] = "directionally_robust_not_independently_validated"
    corrected = stable & pair_results["subtype_adjusted_fdr_bh"].lt(0.10)
    pair_results.loc[corrected, "robustness_class"] = "subtype_adjusted_fdr_support_in_same_dataset"
    pair_results = pair_results.sort_values(
        ["subtype_adjusted_fdr_bh", "subtype_adjusted_beta"], ascending=[True, False]
    )
    pair_results.to_csv(OUT / "hybrid_pair_subtype_loo_robustness.tsv", sep="\t", index=False)

    # Pool different ERBB anchors that share the same partner. This is class-level
    # replication within ASTRAZENECA, with cell-line-clustered uncertainty.
    class_rows = []
    lock_small = lock[["locked_pair_id", "anchor_norm", "partner_norm", "partner"]]
    class_values = values.merge(lock_small, on="locked_pair_id", how="left", validate="many_to_one")
    for partner_norm, group in class_values.groupby("partner_norm", sort=True):
        if group["anchor_norm"].nunique() < 2 or group.loc[group["hybrid_state"], "ModelID"].nunique() < 3:
            continue
        result = adjusted_test(group, cluster=True)
        per_anchor = pair_results[pair_results["partner"].map(lambda x: str(x).casefold()).eq(str(group["partner"].iloc[0]).casefold())]
        class_rows.append({
            "partner": group["partner"].iloc[0],
            "partner_norm": partner_norm,
            "anchors": ";".join(sorted(group["anchor_norm"].unique())),
            "n_anchors": int(group["anchor_norm"].nunique()),
            "n_models": int(group["ModelID"].nunique()),
            "n_hybrid_models": int(group.loc[group["hybrid_state"], "ModelID"].nunique()),
            "positive_anchor_directions": int(per_anchor["discovery_cliffs_delta"].gt(0).sum()),
            "subtype_anchor_adjusted_hybrid_beta": result["beta"],
            "clustered_se": result["se"],
            "clustered_one_sided_p": result["p_one"],
            "design_condition_number": result["condition_number"],
            "estimable": result["estimable"],
        })
    class_results = pd.DataFrame(class_rows)
    if len(class_results):
        class_results["class_level_fdr_bh"] = bh(class_results["clustered_one_sided_p"])
        class_results = class_results.sort_values(["class_level_fdr_bh", "subtype_anchor_adjusted_hybrid_beta"])
    class_results.to_csv(OUT / "hybrid_partner_class_level_robustness.tsv", sep="\t", index=False)
    values.to_csv(OUT / "hybrid_drugcomb_model_subtype_manifest.tsv", sep="\t", index=False)

    focus = pair_results[pair_results["partner"].astype(str).str.casefold().isin({"azd2014", "mk-2206"})]
    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "secondary same-dataset subtype, basal-A, leave-one-out, and ERBB-class robustness analysis",
        "hybrid_pairs_tested": int(len(pair_results)),
        "subtype_adjusted_fdr_lt_0.10": int(pair_results["subtype_adjusted_fdr_bh"].lt(0.10).sum()),
        "basal_A_fdr_lt_0.10": int(pair_results["basal_A_fdr_bh"].lt(0.10).sum()),
        "directionally_robust_pairs": int(stable.sum()),
        "class_level_partners_tested": int(len(class_results)),
        "class_level_fdr_lt_0.10": int(class_results["class_level_fdr_bh"].lt(0.10).sum()) if len(class_results) else 0,
        "focus_pair_robustness": dict(zip(focus["anchor"] + " + " + focus["partner"], focus["robustness_class"])),
        "independent_validation": False,
        "next_gate": "promote only biologically coherent, subtype-robust pairs as ranked hypotheses; seek an independent combination dataset or prospective assay",
    }
    (OUT / "hybrid_drugcomb_robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
