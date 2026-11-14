"""Outcome-free NeoTRIP malignant EMT-state topology and immune coupling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

from analyze_neotrip_spatial_mechanisms import (
    EMT_MARKERS, METRICS, STATES, decode, read_index, read_obs_column,
    read_x_columns, robust_bool, sha256, sign_permutation, bh,
)


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "02_data/01_raw/spatial/NeoTRIP_QUICHE_14290163/nt_preprocessed.h5ad"
CALIBRATION = ROOT / "02_data/03_processed/externally_anchored_emp_v5.5_locked/neotrip_spatial_representation/locked_neotrip_marker_calibration.tsv"
ELIGIBLE = ROOT / "02_data/03_processed/externally_anchored_emp_v5.5_locked/neotrip_spatial_representation/locked_neotrip_patient_spatial_emp.tsv.gz"
PRIOR_SUMMARY = ROOT / "04_analysis/08_biological_landscape/spatial/NeoTRIP_outcome_free/neotrip_spatial_mechanism_summary.json"
PATIENT_STATE = ROOT / "04_analysis/08_biological_landscape/spatial/NeoTRIP_outcome_free/patient_emt_state_spatial_summary.tsv"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/NEOTRIP_SPATIAL_EMT_TOPOLOGY_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/08_biological_landscape/spatial/NeoTRIP_EMT_topology"
SEED = 1272334142
K = 20
MIN_CELLS = 10
EDGE_PSEUDOCOUNT = 0.5
OBS_COLUMNS = ["Patient_ID", "Tissue_ID", "fov", "BiopsyPhase", "isEpithelial"]
STATE_INDEX = {state: i for i, state in enumerate(STATES)}


def process_fov(key, idx, states, coordinates):
    idx = np.asarray(idx, int)
    keep = np.asarray([x in STATE_INDEX for x in states[idx]], bool)
    target = idx[keep]
    if len(target) < MIN_CELLS:
        return None
    local_states = states[target]
    codes = np.asarray([STATE_INDEX[x] for x in local_states], int)
    k = min(K, len(target) - 1)
    tree = cKDTree(coordinates[target])
    _, near = tree.query(coordinates[target], k=k + 1)
    if near.ndim == 1:
        near = near[:, None]
    near = near[:, 1:]
    observed = np.zeros((len(STATES), len(STATES)), float)
    expected = np.zeros_like(observed)
    counts = np.bincount(codes, minlength=len(STATES)).astype(float)
    for source in range(len(STATES)):
        source_rows = np.flatnonzero(codes == source)
        if not len(source_rows):
            continue
        neighbor_codes = codes[near[source_rows]].ravel()
        observed[source] = np.bincount(neighbor_codes, minlength=len(STATES))
        edge_n = len(source_rows) * k
        availability = counts.copy()
        availability[source] -= 1.0
        expected[source] = edge_n * availability / (len(target) - 1)
    return {
        "Patient_ID": str(key[0]), "Tissue_ID": str(key[1]), "fov": str(key[2]),
        "classified_cells": int(len(target)), "k": int(k),
        "state_counts": counts, "observed": observed, "expected": expected,
    }


def symmetric_log_enrichment(observed, expected, a, b):
    i, j = STATE_INDEX[a], STATE_INDEX[b]
    obs = observed[i, j] + observed[j, i]
    exp = expected[i, j] + expected[j, i]
    if exp <= 0:
        return np.nan
    return float(np.log2((obs + EDGE_PSEUDOCOUNT) / (exp + EDGE_PSEUDOCOUNT)))


def corr_test(frame, predictor, outcome, permutations, bootstraps, seed):
    use = frame[[predictor, outcome]].dropna()
    x, y = use[predictor].to_numpy(float), use[outcome].to_numpy(float)
    observed = float(spearmanr(x, y).statistic)
    xr = rankdata(x); yr = rankdata(y)
    xr = xr - xr.mean(); yr = yr - yr.mean()
    denom = np.sqrt(np.dot(xr, xr) * np.dot(yr, yr))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = permutations
    while remaining:
        size = min(1000, remaining)
        perm = np.vstack([rng.permutation(yr) for _ in range(size)])
        null = (perm @ xr) / denom
        exceed += int(np.sum(np.abs(null) >= abs(observed)))
        remaining -= size
    p = float((exceed + 1) / (permutations + 1))
    boot = np.empty(bootstraps, float)
    for b in range(bootstraps):
        idx = rng.integers(0, len(use), len(use))
        boot[b] = spearmanr(x[idx], y[idx]).statistic
    finite = boot[np.isfinite(boot)]
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"predictor": predictor, "metric": outcome, "n": len(use),
            "spearman_rho": observed, "permutation_p": p,
            "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
            "bootstrap_valid": int(len(finite))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.permutations < 10000 or args.bootstrap < 2000:
        raise ValueError("Locked minimum is 10,000 permutations and 2,000 bootstraps")
    if not LOCK.exists():
        raise FileNotFoundError(LOCK)
    prior = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    if prior.get("status") != "PASS" or prior.get("external_outcomes_used") is not False:
        raise RuntimeError("Prior outcome-free spatial analysis is not a clean PASS")
    if prior.get("source_sha256") != sha256(SOURCE):
        raise RuntimeError("NeoTRIP source changed")

    eligible = set(pd.read_csv(ELIGIBLE, sep="\t", usecols=["patient"]).patient.astype(str))
    calibration = pd.read_csv(CALIBRATION, sep="\t").set_index("marker")
    with h5py.File(SOURCE, "r") as handle:
        obs = handle["obs"]
        frame = pd.DataFrame({c: read_obs_column(obs, c) for c in OBS_COLUMNS})
        var_names = pd.Index(read_index(handle["var"]).astype(str))
        missing = [m for m in EMT_MARKERS if m not in var_names]
        if missing:
            raise ValueError(f"Missing frozen EMT markers: {missing}")
        marker_values = read_x_columns(handle, var_names.get_indexer(EMT_MARKERS))
        coordinates = handle["obsm"]["spatial"][...].astype(float)
    frame["isEpithelial"] = robust_bool(frame.isEpithelial)
    keep = frame.Patient_ID.astype(str).isin(eligible) & frame.BiopsyPhase.astype(str).eq("Baseline")
    frame = frame.loc[keep].reset_index(drop=True)
    marker_values, coordinates = marker_values[keep], coordinates[keep]
    for j, marker in enumerate(EMT_MARKERS):
        center = float(calibration.loc[marker, "patient_balanced_center"])
        scale = float(calibration.loc[marker, "patient_balanced_scale"])
        frame[marker + "_frozen_z"] = (marker_values[:, j] - center) / scale
    epithelial = frame.isEpithelial.to_numpy(bool)
    e = frame[[m + "_frozen_z" for m in ["panCK", "CK8/18", "CK5/14"]]].mean(axis=1)
    v = frame["Vimentin_frozen_z"]
    frame["emt_state"] = np.select(
        [epithelial & (e >= 0) & (v < 0), epithelial & (e < 0) & (v >= 0),
         epithelial & (e >= 0) & (v >= 0), epithelial & (e < 0) & (v < 0)],
        STATES, default="non_target")
    states = frame.emt_state.to_numpy(object)

    groups = list(frame.groupby(["Patient_ID", "Tissue_ID", "fov"], sort=False).indices.items())
    fovs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(process_fov)(key, idx, states, coordinates) for key, idx in groups
    )
    fovs = [x for x in fovs if x is not None]
    if not fovs:
        raise RuntimeError("No eligible topology FOVs")

    fov_rows = []
    for item in fovs:
        row = {k: item[k] for k in ["Patient_ID", "Tissue_ID", "fov", "classified_cells", "k"]}
        for i, source in enumerate(STATES):
            row[f"cells::{source}"] = item["state_counts"][i]
            for j, target in enumerate(STATES):
                row[f"observed::{source}::{target}"] = item["observed"][i, j]
                row[f"expected::{source}::{target}"] = item["expected"][i, j]
        fov_rows.append(row)
    fov_table = pd.DataFrame(fov_rows)

    patient_rows = []
    for patient, indices in fov_table.groupby("Patient_ID").groups.items():
        selected = [fovs[i] for i in indices]
        observed = np.sum([x["observed"] for x in selected], axis=0)
        expected = np.sum([x["expected"] for x in selected], axis=0)
        counts = np.sum([x["state_counts"] for x in selected], axis=0)
        total = counts.sum()
        proportions = counts / total
        entropy = -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0])) / np.log(len(STATES))
        eh = symmetric_log_enrichment(observed, expected, "epithelial_dominant", "hybrid_EM")
        hm = symmetric_log_enrichment(observed, expected, "hybrid_EM", "mesenchymal_dominant")
        em = symmetric_log_enrichment(observed, expected, "epithelial_dominant", "mesenchymal_dominant")
        same_obs = float(np.trace(observed)); same_exp = float(np.trace(expected))
        edges = float(observed.sum())
        row = {"Patient_ID": patient, "fovs": len(selected), "classified_cells": int(total),
               "normalized_state_entropy": float(entropy),
               "contact_E_H": eh, "contact_H_M": hm, "contact_E_M": em,
               "hybrid_bridge_contrast": float(0.5 * (eh + hm) - em),
               "same_state_excess_fraction": float((same_obs - same_exp) / edges)}
        for i, state in enumerate(STATES):
            row[f"fraction::{state}"] = float(proportions[i])
            for j, target in enumerate(STATES):
                row[f"enrichment::{state}::{target}"] = (float(np.log2(
                    (observed[i, j] + EDGE_PSEUDOCOUNT) / (expected[i, j] + EDGE_PSEUDOCOUNT)))
                    if expected[i, j] > 0 else np.nan)
        patient_rows.append(row)
    patient = pd.DataFrame(patient_rows)

    primary_specs = [
        ("contact_E_H", "epithelial_hybrid_contact"),
        ("contact_H_M", "hybrid_mesenchymal_contact"),
        ("contact_E_M", "direct_epithelial_mesenchymal_contact"),
        ("hybrid_bridge_contrast", "hybrid_bridge_contrast"),
        ("same_state_excess_fraction", "same_state_excess_fraction"),
    ]
    primary_rows = []
    for index, (column, label) in enumerate(primary_specs):
        values = patient[column].dropna().to_numpy(float)
        effect, p, lo, hi = sign_permutation(values, SEED + index, args.permutations, args.bootstrap)
        primary_rows.append({"estimand": label, "n_patients": len(values), "mean": effect,
                             "permutation_p": p, "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi})
    primary = pd.DataFrame(primary_rows)
    primary["fdr_bh"] = bh(primary.permutation_p)

    state_summary = pd.read_csv(PATIENT_STATE, sep="\t", dtype={"Patient_ID": str})
    overall_rows = []
    for patient_id, group in state_summary.groupby("Patient_ID"):
        weights = group.target_cells.to_numpy(float)
        row = {"Patient_ID": patient_id}
        for metric in METRICS:
            good = group[metric].notna().to_numpy()
            row[metric] = float(np.average(group.loc[good, metric], weights=weights[good])) if good.any() else np.nan
        overall_rows.append(row)
    coupled = patient.merge(pd.DataFrame(overall_rows), on="Patient_ID", how="inner", validate="one_to_one")
    jobs = []
    for p_index, predictor in enumerate(["normalized_state_entropy", "hybrid_bridge_contrast"]):
        for m_index, metric in enumerate(METRICS):
            jobs.append((predictor, metric, SEED + 100 + 100 * p_index + m_index))
    correlations = Parallel(n_jobs=args.n_jobs)(
        delayed(corr_test)(coupled, predictor, metric, args.permutations, args.bootstrap, seed)
        for predictor, metric, seed in jobs
    )
    correlations = pd.DataFrame(correlations)
    correlations["fdr_bh"] = np.nan
    for predictor, idx in correlations.groupby("predictor").groups.items():
        correlations.loc[idx, "fdr_bh"] = bh(correlations.loc[idx, "permutation_p"])

    OUT.mkdir(parents=True, exist_ok=True)
    fov_table.to_csv(OUT / "fov_emt_state_contact_counts.tsv.gz", sep="\t", index=False, compression="gzip")
    patient.to_csv(OUT / "patient_emt_state_topology.tsv", sep="\t", index=False)
    primary.to_csv(OUT / "primary_topology_tests.tsv", sep="\t", index=False)
    correlations.to_csv(OUT / "heterogeneity_immune_coupling.tsv", sep="\t", index=False)

    matrix = np.empty((len(STATES), len(STATES)), float)
    for i, source in enumerate(STATES):
        for j, target in enumerate(STATES):
            matrix[i, j] = patient[f"enrichment::{source}::{target}"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = np.nanmax(np.abs(matrix))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    labels = ["Epithelial", "Mesenchymal", "Hybrid E/M", "Double-low"]
    ax.set_xticks(range(4), labels, rotation=35, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Neighbour state"); ax.set_ylabel("Source state")
    ax.set_title("Patient-mean malignant EMT-state contact enrichment")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > .55 * vmax else "black")
    fig.colorbar(image, ax=ax, label="log2 observed / expected contact")
    fig.tight_layout()
    fig.savefig(OUT / "emt_state_contact_enrichment_heatmap.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "emt_state_contact_enrichment_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "PASS", "analysis_type": "locked outcome-free spatial EMT-state topology",
        "pCR_or_Arm_read": False, "patients": int(patient.Patient_ID.nunique()),
        "eligible_fovs": int(len(fov_table)), "classified_malignant_cells": int(patient.classified_cells.sum()),
        "k_tumor_neighbors": K, "minimum_classified_cells_per_fov": MIN_CELLS,
        "primary_tests": int(len(primary)), "primary_fdr05": primary.loc[primary.fdr_bh < .05, "estimand"].tolist(),
        "coupling_tests": int(len(correlations)),
        "coupling_fdr05": int((correlations.fdr_bh < .05).sum()),
        "interpretation_guardrail": "spatial topology is not temporal lineage tracing",
        "next_gate": "integrate topology with multiscale mechanism and Arm-C response evidence",
    }
    (OUT / "neotrip_emt_topology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
