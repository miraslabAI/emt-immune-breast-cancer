#!/usr/bin/env python
"""Screen the frozen target-first universe for EMT-state drug dependencies."""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK_DIR = ROOT / "04_analysis/11_drug_repositioning/12_target_first_dependency_lock"
LOCK_SUMMARY = LOCK_DIR / "target_first_dependency_lock_summary.json"
UNIVERSE = LOCK_DIR / "frozen_target_first_screen_universe.tsv"
STATES = ROOT / "04_analysis/11_drug_repositioning/10_pharmacogenomic_validation_lock/frozen_ccle_breast_emt_states.tsv"
PRISM_PRIMARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-treatment-info.csv"
PRISM_PRIMARY_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-logfold-change.csv"
PRISM_PRIMARY_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/prism-repurposing-20q2-primary-screen-cell-line-info_breast_models.tsv"
PRISM_SECONDARY_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/secondary_screen/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"
GDSC_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/gdsc/drug_response/GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/GDSC_breast_models.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/13_target_first_dependency_screen"
SEED = 1272244142
N_BOOTSTRAP = 2000
DETERMINATE = {"epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"}
AXIS = {
    "epithelial-enriched": "epithelial_core",
    "hybrid-peak": "em_coactivation",
    "mesenchymal-enriched": "emt_position",
}


def normal_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def bh(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    x = p[good]
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    out[good] = restored
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()) / (len(x) * len(y)))


def bootstrap_difference(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        boot[index] = np.median(xb) - np.median(yb)
    return tuple(float(v) for v in np.quantile(boot, [0.025, 0.975]))


def load_states() -> pd.DataFrame:
    frame = pd.read_csv(STATES, sep="\t", low_memory=False)
    for column in ["epithelial_core", "mesenchymal_core", "emt_position"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["em_coactivation"] = pd.concat([
        frame["epithelial_core"].rank(method="average", pct=True),
        frame["mesenchymal_core"].rank(method="average", pct=True),
    ], axis=1).min(axis=1)
    subtype = frame["ModelSubtypeFeatures"].fillna("").astype(str).str.strip()
    fallback = frame["OncotreeSubtype"].fillna("Unknown").astype(str).str.strip()
    frame["breast_subtype"] = subtype.where(subtype.ne(""), fallback).replace("", "Unknown")
    return frame


def collapse_metadata(group: pd.DataFrame) -> dict[str, object]:
    split = lambda column: ";".join(sorted(set(";".join(group[column].fillna("").astype(str)).split(";")) - {""}))
    return {
        "state_target_genes": split("state_target_genes"),
        "pathway_themes": split("pathway_themes"),
        "molecule_chembl_ids": split("molecule_chembl_ids"),
        "maximum_clinical_phase": float(pd.to_numeric(group["maximum_clinical_phase"], errors="coerce").max()),
        "previous_lincs_shortlist_candidate": group["previous_lincs_shortlist_candidate"].astype(str).str.casefold().eq("true").any(),
    }


def aggregate_response(frame: pd.DataFrame) -> pd.DataFrame:
    # Collapse duplicate screen identifiers/formulations of the same normalized
    # drug name before testing so a model contributes only once per hypothesis.
    keys = ["screen", "state", "normalized_screen_name", "ModelID", "emt_state"]
    annotations = ["epithelial_core", "mesenchymal_core", "emt_position", "em_coactivation", "breast_subtype"]
    aggregations = {
        "sensitivity": "median",
        "source_records": "sum",
        "screen_drug_name": "first",
    }
    aggregations.update({column: "first" for column in annotations})
    aggregations.update({
        "state_target_genes": lambda x: ";".join(sorted(set(";".join(x.fillna("").astype(str)).split(";")) - {""})),
        "pathway_themes": lambda x: ";".join(sorted(set(";".join(x.fillna("").astype(str)).split(";")) - {""})),
        "molecule_chembl_ids": lambda x: ";".join(sorted(set(";".join(x.fillna("").astype(str)).split(";")) - {""})),
        "maximum_clinical_phase": "max",
        "previous_lincs_shortlist_candidate": "max",
    })
    return frame.groupby(keys, as_index=False).agg(aggregations)


def load_primary(universe: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    u = universe[universe["screen"].eq("PRISM_primary")].copy()
    ids = set(u["screen_drug_id"].map(normal_id))
    info = pd.read_csv(PRISM_PRIMARY_INFO, low_memory=False)
    info["screen_drug_id"] = info["broad_id"].map(normal_id)
    info = info[info["screen_drug_id"].isin(ids)]
    header = list(pd.read_csv(PRISM_PRIMARY_RESPONSE, nrows=0).columns)
    id_column = header[0]
    selected = set(info["column_name"]).intersection(header)
    matrix = pd.read_csv(PRISM_PRIMARY_RESPONSE, usecols=lambda c: c == id_column or c in selected, low_memory=False)
    matrix = matrix.rename(columns={id_column: "row_name"})
    long = matrix.melt(id_vars="row_name", var_name="column_name", value_name="raw_response")
    long["raw_response"] = pd.to_numeric(long["raw_response"], errors="coerce")
    long = long.dropna(subset=["raw_response"]).merge(
        info[["column_name", "screen_drug_id"]].drop_duplicates(), on="column_name", how="inner"
    )
    model_map = pd.read_csv(PRISM_PRIMARY_MODELS, sep="\t", low_memory=False)
    model_map = model_map[model_map["depmap_id"].notna()][["row_name", "depmap_id"]].drop_duplicates()
    long = long.merge(model_map, on="row_name", how="inner").merge(
        states, left_on="depmap_id", right_on="ModelID", how="inner"
    ).merge(u, on="screen_drug_id", how="inner")
    long["sensitivity"] = -long["raw_response"]
    long["source_records"] = 1
    return aggregate_response(long)


def load_secondary(universe: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    u = universe[universe["screen"].eq("PRISM_secondary")].copy()
    ids = set(u["screen_drug_id"].map(normal_id))
    blocks = []
    usecols = ["broad_id", "depmap_id", "auc", "r2", "passed_str_profiling"]
    for chunk in pd.read_csv(PRISM_SECONDARY_RESPONSE, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk["screen_drug_id"] = chunk["broad_id"].map(normal_id)
        chunk = chunk[chunk["screen_drug_id"].isin(ids)]
        if len(chunk):
            blocks.append(chunk)
    if not blocks:
        return pd.DataFrame()
    long = pd.concat(blocks, ignore_index=True)
    long["auc"] = pd.to_numeric(long["auc"], errors="coerce")
    long["r2"] = pd.to_numeric(long["r2"], errors="coerce")
    passed = long["passed_str_profiling"].astype(str).str.casefold().eq("true")
    long = long[passed & long["auc"].notna() & long["r2"].ge(0.70)]
    long = long.merge(states, left_on="depmap_id", right_on="ModelID", how="inner").merge(
        u, on="screen_drug_id", how="inner"
    )
    long["sensitivity"] = -long["auc"]
    long["source_records"] = 1
    return aggregate_response(long)


def load_gdsc(universe: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    u = universe[universe["screen"].eq("GDSC2")].copy()
    ids = set(u["screen_drug_id"].map(normal_id))
    response = pd.read_excel(
        GDSC_RESPONSE, sheet_name="Sheet1",
        usecols=["SANGER_MODEL_ID", "DRUG_ID", "LN_IC50"],
    )
    response["screen_drug_id"] = response["DRUG_ID"].map(normal_id)
    response["LN_IC50"] = pd.to_numeric(response["LN_IC50"], errors="coerce")
    response = response[response["screen_drug_id"].isin(ids) & response["LN_IC50"].notna()]
    model_map = pd.read_csv(GDSC_MODELS, sep="\t", low_memory=False)
    model_map = model_map[model_map["BROAD_ID"].notna()][["model_id", "BROAD_ID"]].drop_duplicates()
    response = response.merge(model_map, left_on="SANGER_MODEL_ID", right_on="model_id", how="inner").merge(
        states, left_on="BROAD_ID", right_on="ModelID", how="inner"
    ).merge(u, on="screen_drug_id", how="inner")
    response["sensitivity"] = -response["LN_IC50"]
    response["source_records"] = 1
    return aggregate_response(response)


def adjusted_state_effect(frame: pd.DataFrame, state: str) -> tuple[float, float, float]:
    data = frame.copy()
    data["matched"] = data["emt_state"].eq(state).astype(float)
    subtype = pd.get_dummies(data["breast_subtype"].astype(str), prefix="subtype", drop_first=True, dtype=float)
    design = pd.concat([data[["matched"]].reset_index(drop=True), subtype.reset_index(drop=True)], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    y = data["sensitivity"].to_numpy(float)
    if len(data) < 12 or data["matched"].sum() < 3 or (1 - data["matched"]).sum() < 5:
        return np.nan, np.nan, np.nan
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return np.nan, np.nan, np.nan
    fit = sm.OLS(y, design).fit(cov_type="HC3")
    beta = float(fit.params["matched"])
    p_two = float(fit.pvalues["matched"])
    p_one = p_two / 2 if beta > 0 else 1 - p_two / 2
    return beta, float(fit.bse["matched"]), p_one


def test_universe(response: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, keys in enumerate(response[["screen", "state", "normalized_screen_name"]].drop_duplicates().itertuples(index=False)):
        screen, state, drug = keys
        group = response[
            response["screen"].eq(screen)
            & response["state"].eq(state)
            & response["normalized_screen_name"].eq(drug)
            & response["emt_state"].isin(DETERMINATE)
        ].copy()
        matched = group.loc[group["emt_state"].eq(state), "sensitivity"].to_numpy(float)
        other = group.loc[~group["emt_state"].eq(state), "sensitivity"].to_numpy(float)
        if len(matched) < 3 or len(other) < 5:
            continue
        median_effect = float(np.median(matched) - np.median(other))
        ci_low, ci_high = bootstrap_difference(matched, other, SEED + index)
        delta = cliffs_delta(matched, other)
        mw_p = float(mannwhitneyu(matched, other, alternative="greater", method="auto").pvalue)
        beta, beta_se, adjusted_p = adjusted_state_effect(group, state)
        primary_p = adjusted_p if np.isfinite(adjusted_p) else mw_p
        axis = AXIS[state]
        rho, rho_p = spearmanr(group[axis], group["sensitivity"])
        meta = collapse_metadata(group)
        rows.append({
            "screen": screen,
            "state": state,
            "drug": group["screen_drug_name"].iloc[0],
            "normalized_screen_name": drug,
            "n_matched_state": len(matched),
            "n_other_states": len(other),
            "median_sensitivity_matched": float(np.median(matched)),
            "median_sensitivity_other": float(np.median(other)),
            "matched_minus_other_median": median_effect,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "cliffs_delta": delta,
            "mannwhitney_one_sided_p": mw_p,
            "subtype_adjusted_state_beta": beta,
            "subtype_adjusted_state_se_hc3": beta_se,
            "subtype_adjusted_one_sided_p": adjusted_p,
            "primary_one_sided_p": primary_p,
            "primary_test_used": "subtype_adjusted_OLS_HC3" if np.isfinite(adjusted_p) else "Mann_Whitney_fallback",
            "continuous_axis": axis,
            "axis_spearman_rho": float(rho),
            "axis_spearman_p": float(rho_p),
            **meta,
        })
    result = pd.DataFrame(rows)
    result["primary_fdr_bh_global"] = bh(result["primary_one_sided_p"])
    result["axis_fdr_bh_global"] = bh(result["axis_spearman_p"])
    result["primary_fdr_bh_within_state_screen"] = result.groupby(
        ["state", "screen"], group_keys=False
    )["primary_one_sided_p"].transform(lambda x: bh(x))
    return result


def synthesize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state, drug), group in results.groupby(["state", "normalized_screen_name"], sort=True):
        positive = group["matched_minus_other_median"].gt(0)
        adjusted_positive = group["subtype_adjusted_state_beta"].gt(0)
        global_hits = group["primary_fdr_bh_global"].lt(0.10)
        local_hits = group["primary_fdr_bh_within_state_screen"].lt(0.10)
        strong = group["cliffs_delta"].ge(0.33)
        if global_hits.any() and positive.sum() >= 2:
            evidence = "global_fdr_and_cross_screen_direction"
        elif global_hits.any():
            evidence = "global_fdr_single_screen"
        elif local_hits.any() and positive.sum() >= 2:
            evidence = "screen_fdr_and_cross_screen_direction"
        elif positive.sum() >= 2 and strong.any():
            evidence = "replicated_direction_with_strong_effect"
        elif local_hits.any() or strong.any():
            evidence = "single_screen_signal"
        else:
            evidence = "no_state_selective_dependency_support"
        rows.append({
            "state": state,
            "drug": group.loc[group["primary_one_sided_p"].idxmin(), "drug"],
            "normalized_screen_name": drug,
            "screens_tested": int(group["screen"].nunique()),
            "positive_unadjusted_screens": int(positive.sum()),
            "positive_subtype_adjusted_screens": int(adjusted_positive.sum()),
            "global_fdr_lt_0.10_screens": int(global_hits.sum()),
            "within_state_screen_fdr_lt_0.10_screens": int(local_hits.sum()),
            "strong_effect_screens": int(strong.sum()),
            "best_global_fdr": float(group["primary_fdr_bh_global"].min()),
            "best_cliffs_delta": float(group["cliffs_delta"].max()),
            "dependency_evidence_class": evidence,
            "state_target_genes": ";".join(sorted(set(";".join(group["state_target_genes"]).split(";")) - {""})),
            "pathway_themes": ";".join(sorted(set(";".join(group["pathway_themes"]).split(";")) - {""})),
            "maximum_clinical_phase": float(group["maximum_clinical_phase"].max()),
            "previous_lincs_shortlist_candidate": bool(group["previous_lincs_shortlist_candidate"].any()),
        })
    return pd.DataFrame(rows).sort_values(
        ["state", "global_fdr_lt_0.10_screens", "positive_unadjusted_screens", "best_global_fdr", "best_cliffs_delta"],
        ascending=[True, False, False, True, False],
    )


def main() -> None:
    required = [LOCK_SUMMARY, UNIVERSE, STATES, PRISM_PRIMARY_INFO, PRISM_PRIMARY_RESPONSE,
                PRISM_PRIMARY_MODELS, PRISM_SECONDARY_RESPONSE, GDSC_RESPONSE, GDSC_MODELS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    lock = json.loads(LOCK_SUMMARY.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS" or lock.get("prism_response_values_loaded_or_used") is not False:
        raise RuntimeError("Target-first lock did not pass a clean response firewall")
    universe = pd.read_csv(UNIVERSE, sep="\t", dtype={"screen_drug_id": str}, low_memory=False)
    universe["screen_drug_id"] = universe["screen_drug_id"].map(normal_id)
    states = load_states()
    print("Loading frozen PRISM primary universe")
    primary = load_primary(universe, states)
    print("Loading frozen PRISM secondary universe")
    secondary = load_secondary(universe, states)
    print("Loading frozen GDSC2 universe")
    gdsc = load_gdsc(universe, states)
    response = pd.concat([x for x in [primary, secondary, gdsc] if len(x)], ignore_index=True, sort=False)
    results = test_universe(response)
    synthesis = synthesize(results)

    OUT.mkdir(parents=True, exist_ok=True)
    response.to_csv(OUT / "target_first_breast_model_responses.tsv.gz", sep="\t", index=False, compression="gzip")
    results.to_csv(OUT / "target_first_state_dependency_tests.tsv.gz", sep="\t", index=False, compression="gzip")
    synthesis.to_csv(OUT / "target_first_state_dependency_synthesis.tsv", sep="\t", index=False)
    top = synthesis.groupby("state", as_index=False, group_keys=False).head(50)
    top.to_csv(OUT / "target_first_top50_per_state.tsv", sep="\t", index=False)

    class_counts = synthesis.groupby(["state", "dependency_evidence_class"]).size().rename("drugs").reset_index()
    class_counts.to_csv(OUT / "target_first_evidence_class_counts.tsv", sep="\t", index=False)
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS",
        "analysis_type": "frozen target-first breast EMT-state dependency screen",
        "unique_drug_state_tests": int(len(results)),
        "global_fdr_lt_0.10_tests": int(results["primary_fdr_bh_global"].lt(0.10).sum()),
        "within_state_screen_fdr_lt_0.10_tests": int(results["primary_fdr_bh_within_state_screen"].lt(0.10).sum()),
        "subtype_adjusted_tests": int(results["subtype_adjusted_one_sided_p"].notna().sum()),
        "synthesis_drug_state_pairs": int(len(synthesis)),
        "evidence_classes": {
            f"{row.state}::{row.dependency_evidence_class}": int(row.drugs)
            for row in class_counts.itertuples()
        },
        "primary_multiplicity": "BH across every frozen unique drug-state-screen test",
        "confounding_control": "breast subtype-adjusted OLS with HC3 standard errors; Mann-Whitney fallback only when adjustment not estimable",
        "interpretation_constraint": "selective viability supports tumor-cell dependency but does not establish immune/stromal combination efficacy",
        "next_gate": "integrate target-first dependencies with LINCS repair, compartments and mechanism; construct nonredundant state-specific combination hypotheses",
    }
    (OUT / "target_first_dependency_screen_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
