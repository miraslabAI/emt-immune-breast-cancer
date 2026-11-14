"""Outcome-free replication of the locked TCGA EMT--immune landscape."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from build_tcga_emt_immune_landscape import (
    assign_states, build_axes, heatmap, partial_correlations,
    primary_correlations, zscore,
)


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RANK = ROOT / "02_data/03_processed/analysis_ready_v1/candidate_rank_scores"
CLIN = ROOT / "02_data/03_processed/analysis_ready_v1/harmonized_clinical_covariates"
AUX = ROOT / "02_data/03_processed/bulk_validation"
TCGA_OUT = ROOT / "04_analysis/08_biological_landscape/tcga_emt_immune"
OUT = ROOT / "04_analysis/08_biological_landscape/bulk_replication"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/BULK_EMT_IMMUNE_REPLICATION_LOCK_v1.0.md"

CONFIG = {
    "METABRIC": {
        "rank": RANK / "METABRIC_survival_rank_scores.tsv.gz",
        "clinical": CLIN / "METABRIC_survival_clinical_covariates.tsv",
        "aux": AUX / "METABRIC_scores.tsv.gz",
        "aux_id": "Patient ID",
        "expected_n": 1979,
    },
    "SCANB_GSE96058": {
        "rank": RANK / "SCANB_GSE96058_survival_rank_scores.tsv.gz",
        "clinical": CLIN / "SCANB_GSE96058_survival_clinical_covariates.tsv",
        "aux": AUX / "SCANB_GSE96058_scores.tsv.gz",
        "aux_id": "geo_accession",
        "expected_n": 3273,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_analysis(name: str, cfg: dict, n_boot: int, n_jobs: int):
    raw = pd.read_csv(cfg["rank"], sep="\t")
    if len(raw) != cfg["expected_n"] or raw.sample_id.duplicated().any():
        raise ValueError(f"{name}: expected {cfg['expected_n']} unique samples, got {len(raw)}")
    axes, emt, immune = build_axes(raw)
    clinical = pd.read_csv(cfg["clinical"], sep="\t", usecols=["sample_id", "pam50"], dtype={"sample_id": str})
    aux = pd.read_csv(cfg["aux"], sep="\t", usecols=[cfg["aux_id"], "XCELL_STROMAL_COMPOSITE"],
                      dtype={cfg["aux_id"]: str})
    aux = aux.rename(columns={cfg["aux_id"]: "sample_id", "XCELL_STROMAL_COMPOSITE": "stromal_composite"})
    if aux.sample_id.duplicated().any():
        raise ValueError(f"{name}: duplicated auxiliary sample IDs")
    data = axes.merge(clinical, on="sample_id", how="left", validate="one_to_one")
    data = data.merge(aux, on="sample_id", how="left", validate="one_to_one")
    if data.stromal_composite.notna().sum() != len(data):
        raise ValueError(f"{name}: incomplete xCell stromal linkage")
    data["stromal_z"] = zscore(data["stromal_composite"])
    data["immune_state"] = assign_states(data)
    primary = primary_correlations(data, emt, immune, n_boot, n_jobs)
    partial = partial_correlations(data, emt, immune)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "emt_immune_axes_and_states.tsv.gz", sep="\t", index=False, compression="gzip")
    primary.to_csv(folder / "primary_emt_immune_correlations.tsv", sep="\t", index=False)
    partial.to_csv(folder / "partial_rank_correlations_pam50_stroma.tsv", sep="\t", index=False)
    counts = data.immune_state.value_counts().rename_axis("immune_state").reset_index(name="n")
    counts["proportion"] = counts.n / len(data)
    counts.to_csv(folder / "immune_state_counts.tsv", sep="\t", index=False)
    heatmap(primary, "spearman_rho", emt, immune, f"{name}: locked EMT--immune correlations",
            folder / "primary_correlation_heatmap.png")
    return primary, partial, counts, {
        "rows": len(data), "pam50_available": int(data.pam50.notna().sum()),
        "stromal_available": int(data.stromal_composite.notna().sum()),
        "primary_fdr_lt_0_05": int((primary.fdr_bh < 0.05).sum()),
        "partial_fdr_lt_0_05": int((partial.fdr_bh < 0.05).sum()),
        "immune_state_counts": dict(zip(counts.immune_state, counts.n.astype(int))),
    }


def meta_analysis(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    keys = tables["TCGA_BRCA"][["emt_axis", "immune_axis"]]
    for _, key in keys.iterrows():
        estimates = []
        item = {"emt_axis": key.emt_axis, "immune_axis": key.immune_axis}
        for cohort, table in tables.items():
            hit = table.loc[(table.emt_axis == key.emt_axis) & (table.immune_axis == key.immune_axis)].iloc[0]
            r = float(hit.spearman_rho)
            n = int(hit.n)
            estimates.append((cohort, r, n, float(hit.ci95_low), float(hit.ci95_high)))
            item[f"rho_{cohort}"] = r
            item[f"ci_low_{cohort}"] = float(hit.ci95_low)
            item[f"ci_high_{cohort}"] = float(hit.ci95_high)
        z = np.array([np.arctanh(np.clip(x[1], -0.999999, 0.999999)) for x in estimates])
        w = np.array([x[2] - 3 for x in estimates], float)
        pooled_z = float(np.sum(w * z) / np.sum(w))
        se = float(1.0 / np.sqrt(np.sum(w)))
        pooled_r = float(np.tanh(pooled_z))
        lo, hi = np.tanh([pooled_z - 1.96 * se, pooled_z + 1.96 * se])
        q = float(np.sum(w * (z - pooled_z) ** 2))
        df = len(estimates) - 1
        i2 = float(max(0.0, (q - df) / q) * 100.0) if q > 0 else 0.0
        signs = np.sign([x[1] for x in estimates])
        same_sign = bool(np.all(signs == signs[0]) and signs[0] != 0)
        all_ci_same = bool(all((x[3] > 0 for x in estimates)) or all((x[4] < 0 for x in estimates)))
        item.update({
            "pooled_rho_fixed": pooled_r, "pooled_ci95_low": float(lo), "pooled_ci95_high": float(hi),
            "pooled_p_value": float(2 * norm.sf(abs(pooled_z / se))),
            "same_direction_all_cohorts": same_sign,
            "all_three_bootstrap_ci_exclude_zero_same_direction": all_ci_same,
            "heterogeneity_Q": q, "heterogeneity_df": df,
            "heterogeneity_p": float(chi2.sf(q, df)), "I2_percent": i2,
        })
        rows.append(item)
    out = pd.DataFrame(rows)
    # BH across the 28 locked pooled tests.
    p = out.pooled_p_value.to_numpy()
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    out["pooled_fdr_bh"] = q
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Locked replication requires 1000 bootstrap replicates")
    OUT.mkdir(parents=True, exist_ok=True)
    tcga = pd.read_csv(TCGA_OUT / "primary_emt_immune_correlations.tsv", sep="\t")
    tables = {"TCGA_BRCA": tcga}
    summaries = {}
    for cohort, cfg in CONFIG.items():
        primary, _, _, summary = cohort_analysis(cohort, cfg, args.bootstrap, args.n_jobs)
        tables[cohort] = primary
        summaries[cohort] = summary
    meta = meta_analysis(tables)
    meta.to_csv(OUT / "three_cohort_primary_correlation_meta_analysis.tsv", sep="\t", index=False)
    result = {
        "status": "PASS", "analysis_type": "outcome-free locked biological replication",
        "external_outcomes_used": False, "bootstrap_replicates_per_replication_cohort": args.bootstrap,
        "cohorts": summaries,
        "same_direction_all_three": int(meta.same_direction_all_cohorts.sum()),
        "all_three_ci_exclude_zero_same_direction": int(meta.all_three_bootstrap_ci_exclude_zero_same_direction.sum()),
        "pooled_fdr_lt_0_05": int((meta.pooled_fdr_bh < 0.05).sum()),
        "low_heterogeneity_I2_lt_50": int((meta.I2_percent < 50).sum()),
        "lock": str(LOCK), "lock_sha256": sha256(LOCK),
        "input_sha256": {str(cfg[k]): sha256(cfg[k]) for cfg in CONFIG.values() for k in ["rank", "clinical", "aux"]},
    }
    (OUT / "bulk_replication_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
