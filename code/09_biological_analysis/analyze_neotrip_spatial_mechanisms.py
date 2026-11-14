"""Outcome-free patient-level NeoTRIP EMT-state spatial neighbourhood analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "02_data/01_raw/spatial/NeoTRIP_QUICHE_14290163/nt_preprocessed.h5ad"
CALIBRATION = ROOT / "02_data/03_processed/externally_anchored_emp_v5.5_locked/neotrip_spatial_representation/locked_neotrip_marker_calibration.tsv"
ELIGIBLE = ROOT / "02_data/03_processed/externally_anchored_emp_v5.5_locked/neotrip_spatial_representation/locked_neotrip_patient_spatial_emp.tsv.gz"
AUDIT = ROOT / "04_analysis/01_data_audit/spatial_mechanism/neotrip_outcome_free/neotrip_outcome_free_spatial_design_audit.json"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/NEOTRIP_OUTCOME_FREE_SPATIAL_MECHANISM_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/08_biological_landscape/spatial/NeoTRIP_outcome_free"
SEED = 1272334142
K_NEIGHBORS = 20
PSEUDOCOUNT = 0.01

OBS_COLUMNS = ["Patient_ID", "Tissue_ID", "fov", "BiopsyPhase", "cellAnnotation",
               "cell_cluster_broad", "isEpithelial"]
EMT_MARKERS = ["Vimentin", "panCK", "CK8/18", "CK5/14"]
MARKERS = EMT_MARKERS + ["HLA-ABC", "PD-L1 (SP142)", "PD-L1 (73-10)",
                         "GZMB", "T-bet", "PD-1", "TOX", "CD163", "IDO",
                         "SMA", "PDGFRB", "PDPN", "Calponin"]
STATES = ["epithelial_dominant", "mesenchymal_dominant", "hybrid_EM", "double_low"]
LINEAGES = ["T", "NK", "Mono_Mac", "Stroma"]
METRICS = [
    "neighbor_enrichment::T", "neighbor_enrichment::NK",
    "neighbor_enrichment::Mono_Mac", "neighbor_enrichment::Stroma",
    "neighbor_program::T_cytotoxic", "neighbor_program::T_dysfunction",
    "neighbor_program::MonoMac_suppression", "neighbor_program::Stroma_barrier",
    "tumor_marker::HLA_ABC", "tumor_marker::PDL1",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O", "U"}:
        return np.asarray([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                           for x in arr], dtype=object)
    return arr


def read_index(group):
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return decode(group[key][...])


def read_obs_column(obs, name):
    node = obs[name]
    if isinstance(node, h5py.Dataset):
        return decode(node[...])
    if "codes" in node and "categories" in node:
        codes = node["codes"][...].astype(int)
        cats = decode(node["categories"][...])
        out = np.full(len(codes), None, dtype=object)
        good = codes >= 0
        out[good] = cats[codes[good]]
        return out
    raise ValueError(f"Unsupported obs encoding: {name}")


def read_x_columns(handle, indices):
    node = handle["X"]
    order = np.argsort(indices)
    sorted_idx = np.asarray(indices)[order]
    if isinstance(node, h5py.Dataset):
        values = node[:, sorted_idx]
    else:
        encoding = node.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        shape = tuple(int(x) for x in node.attrs["shape"])
        data, ind, ptr = node["data"][...], node["indices"][...], node["indptr"][...]
        matrix = (sparse.csr_matrix((data, ind, ptr), shape=shape) if encoding == "csr_matrix"
                  else sparse.csc_matrix((data, ind, ptr), shape=shape).tocsr())
        values = matrix[:, sorted_idx].toarray()
    inverse = np.argsort(order)
    return np.asarray(values[:, inverse], dtype=float)


def robust_bool(values):
    return pd.Series(values).astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def grouped_z(values, patients, mask):
    out = np.full(len(values), np.nan, dtype=float)
    for patient in np.unique(patients[mask]):
        idx = np.flatnonzero(mask & (patients == patient))
        x = values[idx].astype(float)
        sd = x.std(ddof=0)
        out[idx] = 0.0 if sd == 0 else (x - x.mean()) / sd
    return out


def weighted_patient_summary(fov_state):
    rows = []
    for (patient, state), group in fov_state.groupby(["Patient_ID", "emt_state"]):
        weights = group.target_cells.to_numpy(float)
        row = {"Patient_ID": patient, "emt_state": state,
               "target_cells": int(weights.sum()), "fovs": int(group.fov.nunique())}
        for metric in METRICS:
            good = group[metric].notna().to_numpy()
            row[metric] = (float(np.average(group.loc[good, metric], weights=weights[good]))
                           if good.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def sign_permutation(delta, seed, permutations, bootstraps):
    delta = np.asarray(delta, float)
    observed = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(delta)))
    null = np.mean(signs * delta, axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
    boot = np.mean(delta[rng.integers(0, len(delta), size=(bootstraps, len(delta)))], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, p, float(lo), float(hi)


def bh(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.permutations < 10000 or args.bootstrap < 2000:
        raise ValueError("Locked analysis requires 10000 permutations and 2000 bootstraps")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("external_outcomes_used") is not False:
        raise RuntimeError("NeoTRIP outcome-free audit is not clean")
    if audit["source_sha256"] != sha256(SOURCE):
        raise RuntimeError("NeoTRIP source changed after audit")

    eligible = set(pd.read_csv(ELIGIBLE, sep="\t", usecols=["patient"]).patient.astype(str))
    calibration = pd.read_csv(CALIBRATION, sep="\t").set_index("marker")
    with h5py.File(SOURCE, "r") as handle:
        obs = handle["obs"]
        frame = pd.DataFrame({c: read_obs_column(obs, c) for c in OBS_COLUMNS})
        var_names = pd.Index(read_index(handle["var"]).astype(str))
        missing = [m for m in MARKERS if m not in var_names]
        if missing:
            raise ValueError(f"Missing locked markers: {missing}")
        marker_values = read_x_columns(handle, var_names.get_indexer(MARKERS))
        spatial = handle["obsm"]["spatial"][...].astype(float)
    frame["isEpithelial"] = robust_bool(frame.isEpithelial)
    keep = (frame.Patient_ID.astype(str).isin(eligible) & frame.BiopsyPhase.astype(str).eq("Baseline"))
    frame = frame.loc[keep].reset_index(drop=True)
    marker_values, spatial = marker_values[keep], spatial[keep]
    for j, marker in enumerate(MARKERS):
        frame[marker] = marker_values[:, j]
    if not np.isfinite(spatial).all():
        raise ValueError("Non-finite spatial coordinates")

    epithelial = frame.isEpithelial.to_numpy(bool)
    for marker in EMT_MARKERS:
        center = float(calibration.loc[marker, "patient_balanced_center"])
        scale = float(calibration.loc[marker, "patient_balanced_scale"])
        frame[marker + "_frozen_z"] = (frame[marker] - center) / scale
    frame["epithelial_strength"] = frame[[m + "_frozen_z" for m in ["panCK", "CK8/18", "CK5/14"]]].mean(axis=1)
    v = frame["Vimentin_frozen_z"]
    e = frame.epithelial_strength
    frame["emt_state"] = np.select(
        [epithelial & (e >= 0) & (v < 0), epithelial & (e < 0) & (v >= 0),
         epithelial & (e >= 0) & (v >= 0), epithelial & (e < 0) & (v < 0)],
        STATES, default="non_target")

    patients = frame.Patient_ID.astype(str).to_numpy()
    broad = frame.cell_cluster_broad.astype(str).to_numpy()
    tmask, mmask, smask = broad == "T", broad == "Mono_Mac", broad == "Stroma"
    marker_z = {}
    for marker in ["GZMB", "T-bet", "PD-1", "TOX"]:
        marker_z[marker] = grouped_z(frame[marker].to_numpy(float), patients, tmask)
    for marker in ["CD163", "PD-L1 (SP142)", "PD-L1 (73-10)", "IDO"]:
        marker_z[marker] = grouped_z(frame[marker].to_numpy(float), patients, mmask)
    for marker in ["SMA", "PDGFRB", "PDPN", "Calponin"]:
        marker_z[marker] = grouped_z(frame[marker].to_numpy(float), patients, smask)
    t_cyt = np.nanmean(np.column_stack([marker_z["GZMB"], marker_z["T-bet"]]), axis=1)
    t_dys = np.nanmean(np.column_stack([marker_z["PD-1"], marker_z["TOX"]]), axis=1)
    mono_sup = np.nanmean(np.column_stack([marker_z[m] for m in ["CD163", "PD-L1 (SP142)", "PD-L1 (73-10)", "IDO"]]), axis=1)
    stroma_barrier = np.nanmean(np.column_stack([marker_z[m] for m in ["SMA", "PDGFRB", "PDPN", "Calponin"]]), axis=1)
    tumor_hla = grouped_z(frame["HLA-ABC"].to_numpy(float), patients, epithelial)
    pdl1a = grouped_z(frame["PD-L1 (SP142)"].to_numpy(float), patients, epithelial)
    pdl1b = grouped_z(frame["PD-L1 (73-10)"].to_numpy(float), patients, epithelial)
    tumor_pdl1 = np.nanmean(np.column_stack([pdl1a, pdl1b]), axis=1)

    groups = list(frame.groupby(["Patient_ID", "Tissue_ID", "fov"], sort=False).indices.items())
    def process_fov(key, idx):
        idx = np.asarray(idx, int)
        target_local = np.flatnonzero(frame.emt_state.iloc[idx].to_numpy() != "non_target")
        if len(target_local) == 0 or len(idx) <= 1:
            return []
        k = min(K_NEIGHBORS + 1, len(idx))
        tree = cKDTree(spatial[idx])
        _, near = tree.query(spatial[idx[target_local]], k=k)
        if near.ndim == 1:
            near = near[:, None]
        near = near[:, 1:]
        global_near = idx[near]
        target_idx = idx[target_local]
        target_states = frame.emt_state.iloc[target_idx].to_numpy()
        rows = []
        base = {lineage: np.mean(broad[idx] == lineage) for lineage in LINEAGES}
        cell_metrics = {}
        for lineage in LINEAGES:
            mask = broad[global_near] == lineage
            frac = mask.mean(axis=1)
            cell_metrics["neighbor_enrichment::" + lineage] = np.log2((frac + PSEUDOCOUNT) / (base[lineage] + PSEUDOCOUNT))
        for label, lineage, values in [
            ("T_cytotoxic", "T", t_cyt), ("T_dysfunction", "T", t_dys),
            ("MonoMac_suppression", "Mono_Mac", mono_sup), ("Stroma_barrier", "Stroma", stroma_barrier)]:
            mask = broad[global_near] == lineage
            vals = values[global_near]
            count = mask.sum(axis=1)
            total = np.nansum(np.where(mask, vals, np.nan), axis=1)
            cell_metrics["neighbor_program::" + label] = np.divide(
                total, count, out=np.full(len(count), np.nan), where=count > 0)
        cell_metrics["tumor_marker::HLA_ABC"] = tumor_hla[target_idx]
        cell_metrics["tumor_marker::PDL1"] = tumor_pdl1[target_idx]
        for state in STATES:
            choose = target_states == state
            if not choose.any():
                continue
            row = {"Patient_ID": str(key[0]), "Tissue_ID": str(key[1]), "fov": str(key[2]),
                   "emt_state": state, "target_cells": int(choose.sum())}
            for metric in METRICS:
                vals = cell_metrics[metric][choose]
                row[metric] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            rows.append(row)
        return rows

    nested = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(process_fov)(key, idx) for key, idx in groups)
    fov_state = pd.DataFrame([row for part in nested for row in part])
    patient_state = weighted_patient_summary(fov_state)

    contrasts, seed = [], SEED
    for state in ["mesenchymal_dominant", "hybrid_EM", "double_low"]:
        for metric in METRICS:
            wide = patient_state.pivot(index="Patient_ID", columns="emt_state", values=metric)
            if state not in wide or "epithelial_dominant" not in wide:
                continue
            pair = wide[[state, "epithelial_dominant"]].dropna()
            if len(pair) < 10:
                contrasts.append({"contrast": f"{state}_minus_epithelial_dominant", "metric": metric,
                                  "paired_patients": len(pair), "mean_difference": np.nan,
                                  "permutation_p": np.nan, "bootstrap_ci95_low": np.nan,
                                  "bootstrap_ci95_high": np.nan, "status": "insufficient_pairs"})
                continue
            effect, p, lo, hi = sign_permutation(
                pair[state] - pair["epithelial_dominant"], seed, args.permutations, args.bootstrap)
            seed += 1
            contrasts.append({"contrast": f"{state}_minus_epithelial_dominant", "metric": metric,
                              "paired_patients": len(pair), "mean_difference": effect,
                              "permutation_p": p, "bootstrap_ci95_low": lo,
                              "bootstrap_ci95_high": hi, "status": "tested"})
    contrasts = pd.DataFrame(contrasts)
    tested = contrasts.status.eq("tested")
    contrasts["fdr_bh"] = np.nan
    contrasts.loc[tested, "fdr_bh"] = bh(contrasts.loc[tested, "permutation_p"])

    OUT.mkdir(parents=True, exist_ok=True)
    state_counts = (frame.loc[frame.emt_state.ne("non_target")]
                    .groupby(["Patient_ID", "emt_state"]).size().rename("cells").reset_index())
    state_counts.to_csv(OUT / "patient_emt_state_cell_counts.tsv", sep="\t", index=False)
    fov_state.to_csv(OUT / "fov_emt_state_spatial_summary.tsv.gz", sep="\t", index=False, compression="gzip")
    patient_state.to_csv(OUT / "patient_emt_state_spatial_summary.tsv", sep="\t", index=False)
    contrasts.to_csv(OUT / "paired_patient_spatial_state_contrasts.tsv", sep="\t", index=False)
    result = {
        "status": "PASS", "analysis_type": "outcome-free NeoTRIP spatial mechanism",
        "external_outcomes_used": False, "forbidden_columns_not_read": ["pCR", "Arm"],
        "eligible_patients": len(eligible), "analyzed_patients": int(frame.Patient_ID.nunique()),
        "fovs": int(frame.fov.nunique()), "cells": int(len(frame)),
        "invasive_epithelial_cells": int(epithelial.sum()),
        "emt_state_cells": {str(k): int(v) for k, v in frame.loc[frame.emt_state.ne("non_target")].emt_state.value_counts().items()},
        "k_neighbors": K_NEIGHBORS, "pseudocount": PSEUDOCOUNT,
        "tested_contrasts": int(tested.sum()), "fdr_lt_0_05": int((contrasts.fdr_bh < 0.05).sum()),
        "permutations": args.permutations, "bootstrap_replicates": args.bootstrap,
        "inference_unit": "paired patient; cells used only to construct neighborhoods",
        "source_sha256": sha256(SOURCE), "calibration_sha256": sha256(CALIBRATION),
        "eligible_patient_file_sha256": sha256(ELIGIBLE),
        "lock": str(LOCK), "lock_sha256": sha256(LOCK),
        "next_step": "integrate bulk, malignant single-cell and spatial evidence into therapy-mechanism matrix",
    }
    (OUT / "neotrip_spatial_mechanism_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
