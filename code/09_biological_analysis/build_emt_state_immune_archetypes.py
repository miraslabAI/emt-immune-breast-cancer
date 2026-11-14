"""Build outcome-free cross-cohort EMT-state immune archetypes."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LAND = ROOT / "04_analysis/08_biological_landscape"
RANK = ROOT / "02_data/03_processed/analysis_ready_v1/candidate_rank_scores"
OUT = LAND / "emt_state_immune_archetypes"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_STATE_IMMUNE_ARCHETYPE_LOCK_v1.0.md"
SEED = 1272324142

COHORTS = {
    "TCGA_BRCA": {
        "axes": LAND / "tcga_emt_immune/tcga_emt_immune_axes_and_states.tsv.gz",
        "rank": RANK / "TCGA_BRCA_survival_rank_scores.tsv.gz",
    },
    "METABRIC": {
        "axes": LAND / "bulk_replication/METABRIC/emt_immune_axes_and_states.tsv.gz",
        "rank": RANK / "METABRIC_survival_rank_scores.tsv.gz",
    },
    "SCANB_GSE96058": {
        "axes": LAND / "bulk_replication/SCANB_GSE96058/emt_immune_axes_and_states.tsv.gz",
        "rank": RANK / "SCANB_GSE96058_survival_rank_scores.tsv.gz",
    },
}

IMMUNE_AXES = [
    "antigen_presentation", "priming_recruitment", "infiltration", "execution",
    "suppressive_recruitment", "overall_activation", "immune_exclusion_pressure",
]

TIP_COMPONENTS = [
    "TIP::Step1", "TIP::Step2.positive", "TIP::Step3.positive",
    "TIP::Step4.T cell.recruiting", "TIP::Step4.CD8 T cell.recruiting",
    "TIP::Step4.Th1 cell.recruiting", "TIP::Step4.Dendritic cell.recruiting",
    "TIP::Step4.NK cell.recruiting", "TIP::Step4.Treg cell.recruiting",
    "TIP::Step4.MDSC.recruiting", "TIP::Step4.Macrophage.recruiting",
    "TIP::Step4.Neutrophil.recruiting", "TIP::Step5.positive",
    "TIP::Step5.negative", "TIP::Step6.positive", "TIP::Step6.negative",
    "TIP::Step7.positive", "TIP::Step7.negative",
]

EMT_STATES = ["epithelial_dominant", "mesenchymal_dominant", "hybrid_EM", "double_low_indeterminate"]
IMMUNE_STATES = ["immune_desert_cold", "immune_excluded", "hot_effective",
                 "inflamed_dysfunctional", "intermediate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def assign_emt_state(frame: pd.DataFrame) -> pd.Series:
    e_hi = frame.epithelial >= frame.epithelial.median()
    m_hi = frame.mesenchymal_core >= frame.mesenchymal_core.median()
    state = pd.Series(index=frame.index, dtype="object")
    state.loc[e_hi & ~m_hi] = "epithelial_dominant"
    state.loc[~e_hi & m_hi] = "mesenchymal_dominant"
    state.loc[e_hi & m_hi] = "hybrid_EM"
    state.loc[~e_hi & ~m_hi] = "double_low_indeterminate"
    if state.isna().any():
        raise ValueError("Incomplete EMT-state assignment")
    return state


def boot_mean(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.mean(values[rng.integers(0, len(values), len(values))])
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def profile_one(cohort, state, feature, values, n_boot, seed):
    x = values.loc[values.emt_state.eq(state), feature].dropna().to_numpy(float)
    lo, hi = boot_mean(x, seed, n_boot)
    return {"cohort": cohort, "emt_state": state, "feature": feature, "n": len(x),
            "mean_z": float(np.mean(x)), "median_z": float(np.median(x)),
            "mean_ci95_low": lo, "mean_ci95_high": hi}


def prepare_cohort(cohort: str, cfg: dict) -> pd.DataFrame:
    axes = pd.read_csv(cfg["axes"], sep="\t", dtype={"sample_id": str})
    raw = pd.read_csv(cfg["rank"], sep="\t", usecols=["sample_id"] + TIP_COMPONENTS,
                      dtype={"sample_id": str})
    data = axes.merge(raw, on="sample_id", how="inner", validate="one_to_one")
    if len(data) != len(axes):
        raise ValueError(f"{cohort}: TIP linkage failure")
    data["cohort"] = cohort
    data["emt_state"] = assign_emt_state(data)
    for c in TIP_COMPONENTS:
        data["component::" + c.removeprefix("TIP::")] = zscore(data[c])
    return data


def enrichment_tables(all_data: pd.DataFrame):
    cohort_rows, pooled_rows = [], []
    for emt in EMT_STATES:
        for immune in IMMUNE_STATES:
            strata = []
            for cohort, group in all_data.groupby("cohort", sort=False):
                a = int(((group.emt_state == emt) & (group.immune_state == immune)).sum())
                b = int(((group.emt_state == emt) & (group.immune_state != immune)).sum())
                c = int(((group.emt_state != emt) & (group.immune_state == immune)).sum())
                d = int(((group.emt_state != emt) & (group.immune_state != immune)).sum())
                odds, p = fisher_exact([[a, b], [c, d]])
                cohort_rows.append({"cohort": cohort, "emt_state": emt, "immune_state": immune,
                                    "a_joint": a, "b_emt_other_immune": b,
                                    "c_other_emt_immune": c, "d_neither": d,
                                    "odds_ratio": odds, "p_value": p})
                strata.append(np.array([[a, b], [c, d]], float))
            fit = StratifiedTable(strata)
            ci_low, ci_high = fit.oddsratio_pooled_confint()
            pooled_rows.append({"emt_state": emt, "immune_state": immune,
                                "common_odds_ratio": float(fit.oddsratio_pooled),
                                "ci95_low": float(ci_low), "ci95_high": float(ci_high),
                                "p_value": float(fit.test_null_odds().pvalue),
                                "heterogeneity_p": float(fit.test_equal_odds().pvalue)})
    cohort_df, pooled_df = pd.DataFrame(cohort_rows), pd.DataFrame(pooled_rows)
    cohort_df["fdr_bh_within_cohort"] = cohort_df.groupby("cohort")["p_value"].transform(bh)
    pooled_df["fdr_bh"] = bh(pooled_df.p_value)
    return cohort_df, pooled_df


def continuous_profiles(all_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group0 in all_data.groupby("cohort", sort=False):
        for axis in ["emt_position", "em_coactivation"]:
            group = group0.copy()
            group["decile"] = pd.qcut(group[axis], 10, labels=False, duplicates="drop") + 1
            for decile, part in group.groupby("decile"):
                for immune in IMMUNE_AXES:
                    rows.append({"cohort": cohort, "emt_axis": axis, "decile": int(decile),
                                 "immune_axis": immune, "n": len(part),
                                 "mean_z": float(part[immune].mean()),
                                 "median_z": float(part[immune].median())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Locked analysis requires 1000 bootstrap replicates")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [prepare_cohort(cohort, cfg) for cohort, cfg in COHORTS.items()]
    all_data = pd.concat(frames, ignore_index=True)
    component_cols = ["component::" + c.removeprefix("TIP::") for c in TIP_COMPONENTS]
    features = IMMUNE_AXES + component_cols
    jobs, seed = [], SEED
    for cohort, frame in all_data.groupby("cohort", sort=False):
        for state in EMT_STATES:
            for feature in features:
                jobs.append((cohort, state, feature, frame, args.bootstrap, seed))
                seed += 1
    profiles = Parallel(n_jobs=args.n_jobs)(delayed(profile_one)(*job) for job in jobs)
    profiles = pd.DataFrame(profiles)
    cohort_enrich, pooled_enrich = enrichment_tables(all_data)
    deciles = continuous_profiles(all_data)
    counts = (all_data.groupby(["cohort", "emt_state", "immune_state"], observed=False)
              .size().rename("n").reset_index())
    counts["within_cohort_proportion"] = counts.n / counts.groupby("cohort").n.transform("sum")

    all_data[["sample_id", "cohort", "emt_state", "immune_state", "epithelial",
              "mesenchymal_core", "emt_position", "em_coactivation"]].to_csv(
        OUT / "sample_emt_and_immune_states.tsv.gz", sep="\t", index=False, compression="gzip")
    profiles.to_csv(OUT / "emt_state_immune_feature_profiles.tsv", sep="\t", index=False)
    counts.to_csv(OUT / "joint_emt_immune_state_counts.tsv", sep="\t", index=False)
    cohort_enrich.to_csv(OUT / "cohort_specific_state_enrichment.tsv", sep="\t", index=False)
    pooled_enrich.to_csv(OUT / "stratified_pooled_state_enrichment.tsv", sep="\t", index=False)
    deciles.to_csv(OUT / "continuous_emt_decile_immune_profiles.tsv", sep="\t", index=False)

    state_counts = {}
    for cohort, frame in all_data.groupby("cohort", sort=False):
        state_counts[cohort] = {
            state: int((frame.emt_state == state).sum()) for state in EMT_STATES
        }
    result = {
        "status": "PASS", "analysis_type": "outcome-free EMT-state immune archetypes",
        "external_outcomes_used": False, "rows": int(len(all_data)),
        "cohort_rows": {k: int(len(v)) for k, v in all_data.groupby("cohort")},
        "emt_state_counts": state_counts,
        "profile_rows": int(len(profiles)), "joint_state_tests": int(len(pooled_enrich)),
        "joint_state_fdr_lt_0_05": int((pooled_enrich.fdr_bh < 0.05).sum()),
        "bootstrap_replicates": args.bootstrap, "random_seed": SEED,
        "lock": str(LOCK), "lock_sha256": sha256(LOCK),
        "input_sha256": {str(p): sha256(p) for cfg in COHORTS.values() for p in cfg.values()},
        "next_step": "cell-of-origin validation in GSE176078 malignant metacells and patient immune composition",
    }
    (OUT / "emt_state_immune_archetype_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
