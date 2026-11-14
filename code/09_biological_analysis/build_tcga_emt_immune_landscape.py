#!/usr/bin/env python
"""Locked, outcome-free TCGA-BRCA EMT--immune landscape analysis."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, rankdata, spearmanr
import statsmodels.api as sm


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RANK_FILE = ROOT / "02_data/03_processed/analysis_ready_v1/candidate_rank_scores/TCGA_BRCA_survival_rank_scores.tsv.gz"
CLIN_FILE = ROOT / "02_data/03_processed/analysis_ready_v1/harmonized_clinical_covariates/TCGA_BRCA_survival_clinical_covariates.tsv"
AUX_FILE = ROOT / "02_data/03_processed/bulk_validation/TCGA_BRCA_scores.tsv.gz"
LOCK_FILE = ROOT / "01_protocol/statistical_analysis_plan/TCGA_EMT_IMMUNE_LANDSCAPE_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/08_biological_landscape/tcga_emt_immune"
SEED = 1272314142

EMP_RAW = [
    "candidate_EMP::EMT-I", "candidate_EMP::EMT-II",
    "candidate_EMP::EMT-III", "candidate_EMP::EMT-IV",
    "candidate_EMP::Hallmark_EMT", "candidate_EMP::Tan2014_E",
    "candidate_EMP::Tan2014_M",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {x.name}: zero/non-finite SD")
    return (x - x.mean()) / sd


def mean_z(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat([zscore(frame[c]) for c in columns], axis=1).mean(axis=1)


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    vals = p[good]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[good] = restored
    return out


def build_axes(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = pd.DataFrame({"sample_id": raw["sample_id"].astype(str)})
    out["epithelial"] = zscore(raw["candidate_EMP::Tan2014_E"])
    out["mesenchymal_core"] = mean_z(raw, [
        "candidate_EMP::Tan2014_M", "candidate_EMP::Hallmark_EMT",
        "candidate_EMP::EMT-II", "candidate_EMP::EMT-III",
    ])
    out["emt_position"] = zscore(out["mesenchymal_core"] - out["epithelial"])
    e_pct = raw["candidate_EMP::Tan2014_E"].rank(method="average", pct=True)
    m_pct = out["mesenchymal_core"].rank(method="average", pct=True)
    out["em_coactivation"] = zscore(pd.concat([e_pct, m_pct], axis=1).min(axis=1))

    out["antigen_presentation"] = mean_z(raw, ["TIP::Step1", "TIP::Step2.positive"])
    out["priming_recruitment"] = mean_z(raw, [
        "TIP::Step3.positive", "TIP::Step4.T cell.recruiting",
        "TIP::Step4.CD8 T cell.recruiting", "TIP::Step4.Th1 cell.recruiting",
        "TIP::Step4.Dendritic cell.recruiting", "TIP::Step4.NK cell.recruiting",
    ])
    out["infiltration"] = zscore(raw["TIP::Step5.positive"]) - zscore(raw["TIP::Step5.negative"])
    out["execution"] = mean_z(raw, ["TIP::Step6.positive", "TIP::Step7.positive"]) - mean_z(
        raw, ["TIP::Step6.negative", "TIP::Step7.negative"]
    )
    out["suppressive_recruitment"] = mean_z(raw, [
        "TIP::Step4.Treg cell.recruiting", "TIP::Step4.MDSC.recruiting",
        "TIP::Step4.Macrophage.recruiting", "TIP::Step4.Neutrophil.recruiting",
    ])
    out["overall_activation"] = pd.concat([
        zscore(out["antigen_presentation"]), zscore(out["priming_recruitment"]),
        zscore(out["infiltration"]), zscore(out["execution"]),
    ], axis=1).mean(axis=1)
    out["immune_exclusion_pressure"] = pd.concat([
        zscore(raw["TIP::Step5.negative"]), zscore(out["suppressive_recruitment"]),
    ], axis=1).mean(axis=1)

    emt = ["epithelial", "mesenchymal_core", "emt_position", "em_coactivation"]
    immune = ["antigen_presentation", "priming_recruitment", "infiltration", "execution",
              "suppressive_recruitment", "overall_activation", "immune_exclusion_pressure"]
    for c in emt + immune:
        out[c] = zscore(out[c])
    return out, emt, immune


def corr_matrix(frame: pd.DataFrame, xs: list[str], ys: list[str]) -> np.ndarray:
    values = frame[xs + ys].to_numpy(float)
    ranks = np.apply_along_axis(rankdata, 0, values)
    cor = np.corrcoef(ranks, rowvar=False)
    return cor[:len(xs), len(xs):]


def bootstrap_once(values: np.ndarray, nx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), len(values))]
    ranks = np.apply_along_axis(rankdata, 0, sampled)
    cor = np.corrcoef(ranks, rowvar=False)
    return cor[:nx, nx:]


def primary_correlations(frame, xs, ys, n_boot, n_jobs):
    observed = corr_matrix(frame, xs, ys)
    pmat = np.empty_like(observed)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pmat[i, j] = spearmanr(frame[x], frame[y]).pvalue
    values = frame[xs + ys].to_numpy(float)
    seeds = np.random.SeedSequence(SEED).generate_state(n_boot)
    boots = np.stack(Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_once)(values, len(xs), int(seed)) for seed in seeds
    ))
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    qmat = bh_fdr(pmat.ravel()).reshape(pmat.shape)
    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            rows.append({"emt_axis": x, "immune_axis": y, "n": len(frame),
                         "spearman_rho": observed[i, j], "ci95_low": lo[i, j],
                         "ci95_high": hi[i, j], "p_value": pmat[i, j], "fdr_bh": qmat[i, j]})
    return pd.DataFrame(rows)


def residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    return sm.OLS(y, design).fit().resid


def partial_correlations(frame, xs, ys):
    needed = xs + ys + ["pam50", "stromal_composite"]
    work = frame[needed].dropna().copy()
    pam = pd.get_dummies(work["pam50"].astype(str), prefix="pam50", drop_first=True, dtype=float)
    cov = pd.concat([zscore(work["stromal_composite"]).rename("stroma"), pam], axis=1)
    design = sm.add_constant(cov.astype(float), has_constant="add").to_numpy(float)
    rows = []
    for x in xs:
        rx = residualize(rankdata(work[x]), design)
        for y in ys:
            ry = residualize(rankdata(work[y]), design)
            result = pearsonr(rx, ry)
            rows.append({"emt_axis": x, "immune_axis": y, "n": len(work),
                         "partial_rank_r": result.statistic, "p_value": result.pvalue,
                         "adjustment": "PAM50 + xCell stromal composite"})
    ans = pd.DataFrame(rows)
    ans["fdr_bh"] = bh_fdr(ans["p_value"])
    return ans


def full_emp_tip(raw: pd.DataFrame) -> pd.DataFrame:
    tips = [c for c in raw.columns if c.startswith("TIP::")]
    rows = []
    for x in EMP_RAW:
        for y in tips:
            r = spearmanr(raw[x], raw[y])
            rows.append({"emp_score": x.removeprefix("candidate_EMP::"),
                         "tip_score": y.removeprefix("TIP::"), "n": len(raw),
                         "spearman_rho": r.statistic, "p_value": r.pvalue})
    ans = pd.DataFrame(rows)
    ans["fdr_bh"] = bh_fdr(ans["p_value"])
    return ans


def subtype_correlations(frame, xs, ys):
    rows = []
    for subtype, group in frame.dropna(subset=["pam50"]).groupby("pam50"):
        if len(group) < 50:
            continue
        for x in xs:
            for y in ys:
                r = spearmanr(group[x], group[y])
                rows.append({"pam50": subtype, "emt_axis": x, "immune_axis": y,
                             "n": len(group), "spearman_rho": r.statistic, "p_value": r.pvalue})
    ans = pd.DataFrame(rows)
    if len(ans):
        ans["fdr_bh_within_subtype"] = ans.groupby("pam50")["p_value"].transform(bh_fdr)
    return ans


def nonlinear_tests(frame, immune):
    work0 = frame.dropna(subset=["pam50", "stromal_composite"]).copy()
    pam = pd.get_dummies(work0["pam50"].astype(str), prefix="pam50", drop_first=True, dtype=float)
    rows = []
    for x in ["emt_position", "em_coactivation"]:
        for y in immune:
            work = pd.DataFrame({"x": zscore(work0[x]), "y": zscore(work0[y]),
                                 "stroma": zscore(work0["stromal_composite"])}, index=work0.index)
            design = pd.concat([work[["x", "stroma"]], (work["x"] ** 2).rename("x_squared"), pam], axis=1)
            fit = sm.OLS(work["y"], sm.add_constant(design.astype(float), has_constant="add")).fit(cov_type="HC3")
            rows.append({"emt_axis": x, "immune_axis": y, "n": len(work),
                         "quadratic_beta": fit.params["x_squared"],
                         "quadratic_se_hc3": fit.bse["x_squared"],
                         "quadratic_p_value": fit.pvalues["x_squared"],
                         "adjustment": "linear EMT + PAM50 + xCell stromal composite"})
    ans = pd.DataFrame(rows)
    ans["quadratic_fdr_bh"] = bh_fdr(ans["quadratic_p_value"])
    return ans


def assign_states(frame: pd.DataFrame) -> pd.Series:
    state = pd.Series("intermediate", index=frame.index, dtype="object")
    hot = ((frame["priming_recruitment"] >= 0) & (frame["infiltration"] >= 0) &
           (frame["execution"] >= 0))
    excluded = ((~hot) & (frame["priming_recruitment"] >= 0) &
                (frame["infiltration"] < 0) & (frame["stromal_z"] >= 0))
    dysfunctional = ((~hot) & (~excluded) & (frame["priming_recruitment"] >= 0) &
                     ((frame["execution"] < 0) | (frame["suppressive_recruitment"] >= 0)))
    cold = ((~hot) & (~excluded) & (~dysfunctional) &
            (frame["priming_recruitment"] < 0) & (frame["overall_activation"] < 0))
    state.loc[hot] = "hot_effective"
    state.loc[excluded] = "immune_excluded"
    state.loc[dysfunctional] = "inflamed_dysfunctional"
    state.loc[cold] = "immune_desert_cold"
    return state


def heatmap(table, value, xs, ys, title, path):
    matrix = table.pivot(index="emt_axis", columns="immune_axis", values=value).reindex(index=xs, columns=ys)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(ys)), labels=ys, rotation=35, ha="right")
    ax.set_yticks(range(len(xs)), labels=xs)
    for i in range(len(xs)):
        for j in range(len(ys)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.bootstrap < 200:
        raise ValueError("At least 200 bootstrap replicates are required; locked run uses 1000.")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RANK_FILE, sep="\t")
    required = ["sample_id"] + EMP_RAW
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing locked columns: {missing}")
    if len(raw) != 1080 or raw["sample_id"].duplicated().any():
        raise ValueError(f"Expected 1080 unique TCGA patients; got {len(raw)} rows")
    score_cols = [c for c in raw.columns if c.startswith(("TIP::", "candidate_EMP::"))]
    if not np.isfinite(raw[score_cols].to_numpy(float)).all():
        raise ValueError("Primary score table contains non-finite values")

    axes, emt, immune = build_axes(raw)
    clinical = pd.read_csv(CLIN_FILE, sep="\t", usecols=["sample_id", "pam50"])
    aux = pd.read_csv(AUX_FILE, sep="\t", usecols=["patient_id", "XCELL_STROMAL_COMPOSITE"])
    aux = aux.rename(columns={"patient_id": "sample_id", "XCELL_STROMAL_COMPOSITE": "stromal_composite"})
    data = axes.merge(clinical, on="sample_id", how="left", validate="one_to_one")
    data = data.merge(aux, on="sample_id", how="left", validate="one_to_one")
    data["stromal_z"] = zscore(data["stromal_composite"])
    data["immune_state"] = assign_states(data)

    primary = primary_correlations(data, emt, immune, args.bootstrap, args.n_jobs)
    partial = partial_correlations(data, emt, immune)
    full = full_emp_tip(raw)
    subtype = subtype_correlations(data, emt, immune)
    nonlinear = nonlinear_tests(data, immune)

    data.to_csv(OUT / "tcga_emt_immune_axes_and_states.tsv.gz", sep="\t", index=False, compression="gzip")
    primary.to_csv(OUT / "primary_emt_immune_correlations.tsv", sep="\t", index=False)
    partial.to_csv(OUT / "partial_rank_correlations_pam50_stroma.tsv", sep="\t", index=False)
    full.to_csv(OUT / "secondary_full_emp_tip_correlations.tsv", sep="\t", index=False)
    subtype.to_csv(OUT / "pam50_stratified_correlations.tsv", sep="\t", index=False)
    nonlinear.to_csv(OUT / "quadratic_nonlinearity_tests.tsv", sep="\t", index=False)
    heatmap(primary, "spearman_rho", emt, immune, "TCGA EMT--immune landscape: primary Spearman correlations",
            OUT / "primary_correlation_heatmap.png")
    heatmap(partial, "partial_rank_r", emt, immune, "TCGA EMT--immune landscape: PAM50/stroma-adjusted",
            OUT / "partial_correlation_heatmap.png")

    counts = data["immune_state"].value_counts().rename_axis("immune_state").reset_index(name="n")
    counts["proportion"] = counts["n"] / len(data)
    counts.to_csv(OUT / "immune_state_counts.tsv", sep="\t", index=False)
    summary = {
        "status": "PASS",
        "analysis_type": "outcome-free biological landscape",
        "external_outcomes_used": False,
        "n_patients": int(len(data)),
        "pam50_available": int(data["pam50"].notna().sum()),
        "stromal_composite_available": int(data["stromal_composite"].notna().sum()),
        "bootstrap_replicates": args.bootstrap,
        "random_seed": SEED,
        "primary_tests": int(len(primary)),
        "primary_fdr_lt_0_05": int((primary["fdr_bh"] < 0.05).sum()),
        "partial_tests": int(len(partial)),
        "partial_fdr_lt_0_05": int((partial["fdr_bh"] < 0.05).sum()),
        "immune_state_counts": dict(zip(counts["immune_state"], counts["n"].astype(int))),
        "inputs_sha256": {str(p): sha256(p) for p in [RANK_FILE, CLIN_FILE, AUX_FILE, LOCK_FILE]},
        "limitations": [
            "Bulk-transcriptomic EMT axes are state/coactivation proxies, not temporal trajectories.",
            "No direct tumor-purity estimate was available in the locked inputs; xCell stromal composite is a sensitivity covariate.",
            "Immune-state labels are descriptive, threshold-based phenotypes and are not treatment-response classifiers.",
        ],
    }
    (OUT / "tcga_emt_immune_landscape_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
