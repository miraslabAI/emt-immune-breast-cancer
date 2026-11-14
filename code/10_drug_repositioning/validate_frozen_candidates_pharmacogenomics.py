#!/usr/bin/env python
"""Validate frozen state-specific candidates in PRISM and GDSC2.

Only candidates and cell states frozen before response access are tested.
Higher standardized sensitivity always means greater drug sensitivity.
Selective viability is supportive evidence, not a mandatory property of a
non-cytotoxic microenvironment-remodelling agent.
"""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK_DIR = ROOT / "04_analysis/11_drug_repositioning/10_pharmacogenomic_validation_lock"
LOCK_SUMMARY = LOCK_DIR / "pharmacogenomic_validation_lock_summary.json"
FROZEN = LOCK_DIR / "frozen_lincs_candidates_for_viability.tsv"
STATES = LOCK_DIR / "frozen_ccle_breast_emt_states.tsv"
PRISM_PRIMARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-treatment-info.csv"
PRISM_PRIMARY_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-logfold-change.csv"
PRISM_PRIMARY_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/prism-repurposing-20q2-primary-screen-cell-line-info_breast_models.tsv"
PRISM_SECONDARY_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/secondary_screen/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"
GDSC_RESPONSE = ROOT / "02_data/01_raw/drug_sensitivity/gdsc/drug_response/GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/GDSC_breast_models.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/11_pharmacogenomic_validation"
SEED = 1272234142
N_BOOTSTRAP = 5000
DETERMINATE_STATES = {"epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"}
AXIS_BY_STATE = {
    "epithelial-enriched": "epithelial_core",
    "hybrid-peak": "em_coactivation",
    "mesenchymal-enriched": "emt_position",
}


def normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def bh(pvalues: pd.Series) -> np.ndarray:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    values = p[good]
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[good] = restored
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()) / (len(x) * len(y)))


def bootstrap_median_difference(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        values[index] = np.median(xb) - np.median(yb)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def prepare_states() -> pd.DataFrame:
    states = pd.read_csv(STATES, sep="\t", low_memory=False)
    for column in ["epithelial_core", "mesenchymal_core", "emt_position"]:
        states[column] = pd.to_numeric(states[column], errors="coerce")
    e_pct = states["epithelial_core"].rank(method="average", pct=True)
    m_pct = states["mesenchymal_core"].rank(method="average", pct=True)
    states["em_coactivation"] = pd.concat([e_pct, m_pct], axis=1).min(axis=1)
    return states


def frozen_candidates() -> pd.DataFrame:
    frame = pd.read_csv(FROZEN, sep="\t", low_memory=False)
    frame["normalized_name"] = frame["pert_iname"].map(normal_name)
    return frame.drop_duplicates(["state", "normalized_name"])


def load_prism_primary(states: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    names = set(candidates["normalized_name"])
    info = pd.read_csv(PRISM_PRIMARY_INFO, low_memory=False)
    info["normalized_name"] = info["name"].map(normal_name)
    info = info[info["normalized_name"].isin(names)].copy()
    selected_columns = set(info["column_name"].dropna().astype(str))
    header = list(pd.read_csv(PRISM_PRIMARY_RESPONSE, nrows=0).columns)
    id_column = header[0]
    present = selected_columns.intersection(header)
    response = pd.read_csv(
        PRISM_PRIMARY_RESPONSE,
        usecols=lambda column: column == id_column or column in present,
        low_memory=False,
    ).rename(columns={id_column: "row_name"})
    long = response.melt(id_vars="row_name", var_name="column_name", value_name="raw_response")
    long["raw_response"] = pd.to_numeric(long["raw_response"], errors="coerce")
    long = long.dropna(subset=["raw_response"])
    long = long.merge(
        info[["column_name", "normalized_name", "name", "broad_id", "dose"]].drop_duplicates(),
        on="column_name", how="inner",
    )
    model_map = pd.read_csv(PRISM_PRIMARY_MODELS, sep="\t", low_memory=False)
    model_map = model_map[model_map["depmap_id"].notna()][["row_name", "depmap_id"]].drop_duplicates()
    long = long.merge(model_map, on="row_name", how="inner")
    long = long.merge(states, left_on="depmap_id", right_on="ModelID", how="inner")
    long["sensitivity"] = -long["raw_response"]
    result = long.groupby(
        ["normalized_name", "ModelID", "emt_state"], as_index=False
    ).agg(
        sensitivity=("sensitivity", "median"),
        epithelial_core=("epithelial_core", "first"),
        mesenchymal_core=("mesenchymal_core", "first"),
        emt_position=("emt_position", "first"),
        em_coactivation=("em_coactivation", "first"),
        source_records=("raw_response", "size"),
    )
    result["screen"] = "PRISM_primary"
    result["response_metric"] = "negative_logfold_change"
    return result


def load_prism_secondary(states: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    names = set(candidates["normalized_name"])
    blocks = []
    usecols = ["broad_id", "depmap_id", "name", "auc", "ic50", "r2", "passed_str_profiling"]
    for chunk in pd.read_csv(PRISM_SECONDARY_RESPONSE, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk["normalized_name"] = chunk["name"].map(normal_name)
        chunk = chunk[chunk["normalized_name"].isin(names)]
        if len(chunk):
            blocks.append(chunk)
    if not blocks:
        return pd.DataFrame()
    response = pd.concat(blocks, ignore_index=True)
    response["auc"] = pd.to_numeric(response["auc"], errors="coerce")
    response["ic50"] = pd.to_numeric(response["ic50"], errors="coerce")
    response["r2"] = pd.to_numeric(response["r2"], errors="coerce")
    passed = response["passed_str_profiling"].astype(str).str.casefold().eq("true")
    response = response[passed & response["auc"].notna() & response["r2"].ge(0.70)].copy()
    response = response.merge(states, left_on="depmap_id", right_on="ModelID", how="inner")
    response["sensitivity"] = -response["auc"]
    result = response.groupby(
        ["normalized_name", "ModelID", "emt_state"], as_index=False
    ).agg(
        sensitivity=("sensitivity", "median"),
        epithelial_core=("epithelial_core", "first"),
        mesenchymal_core=("mesenchymal_core", "first"),
        emt_position=("emt_position", "first"),
        em_coactivation=("em_coactivation", "first"),
        source_records=("auc", "size"),
        median_ic50=("ic50", "median"),
        median_curve_r2=("r2", "median"),
    )
    result["screen"] = "PRISM_secondary"
    result["response_metric"] = "negative_auc"
    return result


def load_gdsc(states: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    names = set(candidates["normalized_name"])
    usecols = ["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC", "RMSE"]
    response = pd.read_excel(GDSC_RESPONSE, sheet_name="Sheet1", usecols=usecols)
    response["normalized_name"] = response["DRUG_NAME"].map(normal_name)
    response = response[response["normalized_name"].isin(names)].copy()
    for column in ["LN_IC50", "AUC", "RMSE"]:
        response[column] = pd.to_numeric(response[column], errors="coerce")
    model_map = pd.read_csv(GDSC_MODELS, sep="\t", low_memory=False)
    model_map = model_map[model_map["BROAD_ID"].notna()][["model_id", "BROAD_ID"]].drop_duplicates()
    response = response.merge(model_map, left_on="SANGER_MODEL_ID", right_on="model_id", how="inner")
    response = response.merge(states, left_on="BROAD_ID", right_on="ModelID", how="inner")
    response = response[response["LN_IC50"].notna()].copy()
    response["sensitivity"] = -response["LN_IC50"]
    result = response.groupby(
        ["normalized_name", "ModelID", "emt_state"], as_index=False
    ).agg(
        sensitivity=("sensitivity", "median"),
        epithelial_core=("epithelial_core", "first"),
        mesenchymal_core=("mesenchymal_core", "first"),
        emt_position=("emt_position", "first"),
        em_coactivation=("em_coactivation", "first"),
        source_records=("LN_IC50", "size"),
        median_auc=("AUC", "median"),
        median_rmse=("RMSE", "median"),
    )
    result["screen"] = "GDSC2"
    result["response_metric"] = "negative_ln_ic50"
    return result


def test_candidates(response: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test_index = 0
    for candidate in candidates.itertuples(index=False):
        subset = response[
            response["normalized_name"].eq(candidate.normalized_name)
            & response["emt_state"].isin(DETERMINATE_STATES)
        ].copy()
        if subset.empty:
            continue
        matched = subset.loc[subset["emt_state"].eq(candidate.state), "sensitivity"].dropna().to_numpy(float)
        other = subset.loc[~subset["emt_state"].eq(candidate.state), "sensitivity"].dropna().to_numpy(float)
        if len(matched) < 3 or len(other) < 5:
            status = "insufficient_state_sample_size"
            effect = delta = pvalue = ci_low = ci_high = np.nan
        else:
            status = "tested"
            effect = float(np.median(matched) - np.median(other))
            delta = cliffs_delta(matched, other)
            pvalue = float(mannwhitneyu(matched, other, alternative="greater", method="auto").pvalue)
            ci_low, ci_high = bootstrap_median_difference(matched, other, SEED + test_index)
        axis = AXIS_BY_STATE[candidate.state]
        finite = subset[[axis, "sensitivity"]].dropna()
        rho, rho_p = (np.nan, np.nan)
        if len(finite) >= 8 and finite[axis].nunique() > 1 and finite["sensitivity"].nunique() > 1:
            rho, rho_p = spearmanr(finite[axis], finite["sensitivity"])
        rows.append({
            "screen": subset["screen"].iloc[0],
            "response_metric": subset["response_metric"].iloc[0],
            "state": candidate.state,
            "candidate": candidate.pert_iname,
            "normalized_name": candidate.normalized_name,
            "validation_role": candidate.validation_role,
            "status": status,
            "n_matched_state": len(matched),
            "n_other_states": len(other),
            "median_sensitivity_matched": float(np.median(matched)) if len(matched) else np.nan,
            "median_sensitivity_other": float(np.median(other)) if len(other) else np.nan,
            "matched_minus_other_median": effect,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "cliffs_delta": delta,
            "one_sided_mannwhitney_p": pvalue,
            "continuous_axis": axis,
            "axis_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "axis_spearman_p": float(rho_p) if np.isfinite(rho_p) else np.nan,
            "lincs_evidence_adjusted_score": candidate.evidence_adjusted_score,
            "lincs_direction_compatible_chembl_match": candidate.direction_compatible_chembl_match,
            "lincs_mechanism_genes": candidate.mechanism_genes,
        })
        test_index += 1
    result = pd.DataFrame(rows)
    if len(result):
        result["mannwhitney_fdr_bh_global"] = bh(result["one_sided_mannwhitney_p"])
        result["axis_spearman_fdr_bh_global"] = bh(result["axis_spearman_p"])
    return result


def synthesize(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in candidates.itertuples(index=False):
        group = results[
            results["state"].eq(candidate.state)
            & results["normalized_name"].eq(candidate.normalized_name)
            & results["status"].eq("tested")
        ]
        positive = int(group["matched_minus_other_median"].gt(0).sum())
        strong = int(group["cliffs_delta"].ge(0.33).sum())
        nominal = int(group["one_sided_mannwhitney_p"].lt(0.05).sum())
        fdr = int(group["mannwhitney_fdr_bh_global"].lt(0.10).sum())
        if positive >= 2 and (nominal >= 1 or strong >= 2):
            classification = "replicated_state_selective_sensitivity"
        elif nominal >= 1 or strong >= 1:
            classification = "single_screen_state_selective_support"
        elif positive >= 2:
            classification = "replicated_directional_sensitivity_only"
        else:
            classification = "no_state_selective_viability_support"
        rows.append({
            "state": candidate.state,
            "candidate": candidate.pert_iname,
            "normalized_name": candidate.normalized_name,
            "validation_role": candidate.validation_role,
            "screens_tested": int(len(group)),
            "positive_effect_screens": positive,
            "strong_effect_screens_cliffs_delta_ge_0.33": strong,
            "nominal_p_lt_0.05_screens": nominal,
            "global_fdr_lt_0.10_screens": fdr,
            "viability_evidence_class": classification,
            "lincs_evidence_adjusted_score": candidate.evidence_adjusted_score,
            "direction_compatible_chembl_match": candidate.direction_compatible_chembl_match,
            "mechanism_genes": candidate.mechanism_genes,
            "interpretation_constraint": (
                "absence of selective killing does not exclude a non-cytotoxic state-remodelling role"
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    required = [LOCK_SUMMARY, FROZEN, STATES, PRISM_PRIMARY_INFO, PRISM_PRIMARY_RESPONSE,
                PRISM_PRIMARY_MODELS, PRISM_SECONDARY_RESPONSE, GDSC_RESPONSE, GDSC_MODELS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    lock = json.loads(LOCK_SUMMARY.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS":
        raise RuntimeError("Pharmacogenomic validation lock did not pass")
    if lock.get("prism_response_values_loaded_or_used") is not False or lock.get("gdsc_response_values_loaded_or_used") is not False:
        raise RuntimeError("Outcome firewall was not clean at candidate freeze")

    states = prepare_states()
    candidates = frozen_candidates()
    print("Loading frozen PRISM primary candidates")
    primary = load_prism_primary(states, candidates)
    print("Loading frozen PRISM secondary candidates")
    secondary = load_prism_secondary(states, candidates)
    print("Loading frozen GDSC2 candidates")
    gdsc = load_gdsc(states, candidates)
    response = pd.concat([x for x in [primary, secondary, gdsc] if len(x)], ignore_index=True, sort=False)
    if response.empty:
        raise RuntimeError("No frozen candidate response values linked to breast models")

    results = pd.concat(
        [test_candidates(group, candidates) for _, group in response.groupby("screen", sort=True)],
        ignore_index=True,
    )
    # Recalculate multiplicity globally after concatenating all screens.
    results["mannwhitney_fdr_bh_global"] = bh(results["one_sided_mannwhitney_p"])
    results["axis_spearman_fdr_bh_global"] = bh(results["axis_spearman_p"])
    synthesis = synthesize(results, candidates)

    OUT.mkdir(parents=True, exist_ok=True)
    response.to_csv(OUT / "frozen_candidate_breast_model_responses.tsv.gz", sep="\t", index=False, compression="gzip")
    results.to_csv(OUT / "state_specific_pharmacogenomic_tests.tsv", sep="\t", index=False)
    synthesis.to_csv(OUT / "state_specific_pharmacogenomic_synthesis.tsv", sep="\t", index=False)
    coverage = response.groupby(["screen", "normalized_name"], as_index=False).agg(
        breast_models=("ModelID", "nunique"),
        determinate_models=("emt_state", lambda x: int(pd.Series(x).isin(DETERMINATE_STATES).sum())),
    )
    coverage.to_csv(OUT / "tested_candidate_model_coverage.tsv", sep="\t", index=False)

    class_counts = synthesis["viability_evidence_class"].value_counts().to_dict()
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS",
        "analysis_type": "locked state-specific breast pharmacogenomic validation",
        "screens": sorted(response["screen"].unique()),
        "frozen_candidate_state_pairs": int(len(candidates)),
        "candidate_state_screen_tests": int(len(results)),
        "tested_results": int(results["status"].eq("tested").sum()),
        "global_fdr_lt_0.10": int(results["mannwhitney_fdr_bh_global"].lt(0.10).sum()),
        "viability_evidence_classes": {str(k): int(v) for k, v in class_counts.items()},
        "response_direction": "higher sensitivity values always indicate greater drug sensitivity",
        "primary_test": "one-sided matched EMT state greater sensitivity than other determinate breast EMT states",
        "continuous_confirmation": "Spearman correlation with epithelial, E/M coactivation, or EMT-position axis as prespecified",
        "interpretation_constraint": "viability support strengthens a tumor-cell dependency claim; null viability does not refute immune/stromal remodelling",
        "next_gate": "integrate LINCS repair, compartment/mechanism and viability evidence; then nominate complementary combination partners per EMT state",
    }
    (OUT / "pharmacogenomic_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
