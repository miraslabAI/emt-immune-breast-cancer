"""Locked patient-level and malignant-metacell EMT--immune analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Prevent each of the 12 patient-level workers from creating its own pool of
# BLAS/OpenMP threads.  This also avoids the Intel/OpenMP runtime collision seen
# when the CUDA/PyTorch backend is loaded in the current conda environment.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import pearsonr, rankdata, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "02_data/03_processed/single_cell/GSE176078"
H5AD = DATA / "GSE176078_malignant_patient_metacells.h5ad"
META = DATA / "GSE176078_metacell_metadata_scores.tsv"
COMPOSITION = DATA / "GSE176078_patient_cell_composition.tsv"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/GSE176078_CELL_ORIGIN_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/08_biological_landscape/single_cell/GSE176078_cell_origin"
SEED = 1272329142

PROGRAMS = {
    "antigen_presentation": ["B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "NLRC5", "PSMB8", "PSMB9"],
    "ifn_response": ["STAT1", "IRF1", "GBP1", "CXCL9", "CXCL10", "CXCL11", "IDO1", "IFIT1", "IFIT3", "ISG15"],
    "checkpoint_ligands": ["CD274", "PDCD1LG2", "LGALS9", "PVR", "NECTIN2", "CD80", "CD86"],
    "t_nk_recruiting": ["CXCL9", "CXCL10", "CXCL11", "CCL5", "CCL4", "XCL1", "XCL2"],
    "myeloid_recruiting": ["CCL2", "CSF1", "CSF2", "CXCL1", "CXCL2", "CXCL3", "CXCL5", "CXCL8", "CCL20"],
    "tgfb_ecm": ["TGFB1", "TGFB2", "TGFBI", "SERPINE1", "CTGF", "FN1", "COL1A1", "COL1A2", "ITGA5", "LOX"],
}

PATIENT_EMT = ["mean_emt_position", "mesenchymal_occupancy", "hybrid_occupancy",
               "emt_position_sd", "emt_state_entropy"]
COMPOSITION_FEATURES = ["T-cells", "Myeloid", "CAFs", "Endothelial", "B-cells", "PVL", "T_to_myeloid_logratio"]
EMT_STATES = ["epithelial_dominant", "mesenchymal_dominant", "hybrid_EM", "double_low_indeterminate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def weighted_mean(x, w):
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def weighted_sd(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def _decode_strings(values):
    return pd.Index([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ])


def read_h5ad_matrix(path):
    """Read only X/obs_names/var_names without importing anndata/PyTorch."""
    with h5py.File(path, "r") as handle:
        obs_key = handle["obs"].attrs.get("_index", "_index")
        var_key = handle["var"].attrs.get("_index", "_index")
        if isinstance(obs_key, bytes):
            obs_key = obs_key.decode("utf-8")
        if isinstance(var_key, bytes):
            var_key = var_key.decode("utf-8")
        obs_names = _decode_strings(handle["obs"][obs_key][...])
        var_names = _decode_strings(handle["var"][var_key][...])
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            matrix = sparse.csr_matrix(node[...])
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            shape = tuple(int(v) for v in node.attrs["shape"])
            data = node["data"][...]
            indices = node["indices"][...]
            indptr = node["indptr"][...]
            if encoding == "csr_matrix":
                matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
            elif encoding == "csc_matrix":
                matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Unsupported H5AD X encoding: {encoding!r}")
    return matrix, obs_names, var_names


def program_scores(x, obs_names, var_names, metadata):
    names = pd.Index(var_names.astype(str).str.upper())
    if names.duplicated().any():
        raise ValueError("Duplicated upper-case gene symbols in metacell h5ad")
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    else:
        x = x.tocsr()
    if x.data.size and np.nanmin(x.data) < 0:
        raise ValueError("Expected non-negative metacell counts")
    library = np.asarray(x.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("Zero-library metacell")
    coverage, scores = [], pd.DataFrame(index=obs_names.astype(str))
    norm = sparse.diags(1e6 / library) @ x
    norm.data = np.log1p(norm.data)
    for program, genes in PROGRAMS.items():
        present = [g for g in genes if g in names]
        missing = [g for g in genes if g not in names]
        coverage.append({"program": program, "requested": len(genes), "detected": len(present),
                         "fraction": len(present) / len(genes), "detected_genes": ";".join(present),
                         "missing_genes": ";".join(missing)})
        if len(present) < 3:
            raise ValueError(f"Insufficient coverage for {program}: {present}")
        idx = names.get_indexer(present)
        values = norm[:, idx].toarray().astype(float)
        sd = values.std(axis=0, ddof=0)
        keep = sd > 0
        if keep.sum() < 3:
            raise ValueError(f"Insufficient variable genes for {program}")
        z = (values[:, keep] - values[:, keep].mean(axis=0)) / sd[keep]
        scores["program::" + program] = z.mean(axis=1)
    scores.index.name = "metacell_id"
    scores = scores.reset_index()
    return metadata.merge(scores, on="metacell_id", validate="one_to_one"), pd.DataFrame(coverage)


def assign_states(meta):
    meta = meta.copy()
    meta["E_subtype_z"] = meta.groupby("subtype")["E_Tan2014_cellline"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    meta["M_subtype_z"] = meta.groupby("subtype")["M_Tan2014_cellline"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    meta["emt_position_subtype"] = meta.M_subtype_z - meta.E_subtype_z
    e_hi = meta.E_Tan2014_cellline >= meta.groupby("subtype").E_Tan2014_cellline.transform("median")
    m_hi = meta.M_Tan2014_cellline >= meta.groupby("subtype").M_Tan2014_cellline.transform("median")
    meta["emt_state"] = np.select(
        [e_hi & ~m_hi, ~e_hi & m_hi, e_hi & m_hi, ~e_hi & ~m_hi], EMT_STATES, default="ERROR")
    meta["emt_position_within_patient"] = meta.M_within_patient_z - meta.E_within_patient_z
    e_pct = meta.groupby("patient_id").E_Tan2014_cellline.rank(pct=True)
    m_pct = meta.groupby("patient_id").M_Tan2014_cellline.rank(pct=True)
    meta["coactivation_within_patient"] = np.minimum(e_pct, m_pct)
    return meta


def patient_features(meta):
    rows = []
    program_cols = ["program::" + x for x in PROGRAMS]
    for patient, group in meta.groupby("patient_id"):
        w = group.n_cells.to_numpy(float)
        probs = np.array([(w[group.emt_state.eq(s)].sum() / w.sum()) for s in EMT_STATES])
        nz = probs[probs > 0]
        row = {"patient_id": patient, "subtype": group.subtype.iloc[0],
               "malignant_cells": int(w.sum()), "metacells": len(group),
               "mean_emt_position": weighted_mean(group.emt_position_subtype, w),
               "mesenchymal_occupancy": float(probs[1]), "hybrid_occupancy": float(probs[2]),
               "emt_position_sd": weighted_sd(group.emt_position_subtype, w),
               "emt_state_entropy": float(-(nz * np.log(nz)).sum() / np.log(len(EMT_STATES)))}
        for state, prob in zip(EMT_STATES, probs):
            row["occupancy::" + state] = float(prob)
        for col in program_cols:
            row[col] = weighted_mean(group[col], w)
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_bootstrap(seed, x, y, permutations, bootstraps):
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    perm = np.empty(permutations)
    for i in range(permutations):
        perm[i] = spearmanr(x, rng.permutation(y)).statistic
    p = float((1 + np.sum(np.abs(perm) >= abs(observed))) / (permutations + 1))
    boot = []
    for _ in range(bootstraps):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boot.append(r)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, p, float(lo), float(hi)


def partial_subtype(x, y, subtype):
    cov = pd.get_dummies(pd.Series(subtype).astype(str), drop_first=True, dtype=float)
    design = sm.add_constant(cov, has_constant="add").to_numpy(float)
    rx = sm.OLS(rankdata(x), design).fit().resid
    ry = sm.OLS(rankdata(y), design).fit().resid
    result = pearsonr(rx, ry)
    return float(result.statistic), float(result.pvalue)


def patient_correlations(patients, xcols, ycols, permutations, bootstraps, n_jobs, seed0):
    jobs = []
    seed = seed0
    for x in xcols:
        for y in ycols:
            jobs.append((seed, x, y)); seed += 1
    def one(seed, x, y):
        frame = patients[[x, y, "subtype"]].dropna()
        r, p, lo, hi = permutation_bootstrap(seed, frame[x], frame[y], permutations, bootstraps)
        pr, pp = partial_subtype(frame[x].to_numpy(), frame[y].to_numpy(), frame.subtype.to_numpy())
        return {"emt_feature": x, "target": y, "n_patients": len(frame), "spearman_rho": r,
                "permutation_p": p, "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
                "subtype_adjusted_partial_rank_r": pr, "subtype_adjusted_p": pp}
    out = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(*job) for job in jobs))
    out["permutation_fdr_bh"] = bh(out.permutation_p)
    out["subtype_adjusted_fdr_bh"] = bh(out.subtype_adjusted_p)
    return out


def within_patient_models(meta):
    rows = []
    for program in PROGRAMS:
        outcome = "program::" + program
        formula = (f'Q("{outcome}") ~ emt_position_within_patient + coactivation_within_patient + '
                   'cell_cycle_rank + stress_rank + C(patient_id)')
        fit = smf.ols(formula, data=meta).fit(cov_type="cluster", cov_kwds={"groups": meta.patient_id})
        for term in ["emt_position_within_patient", "coactivation_within_patient"]:
            rows.append({"program": program, "term": term, "n_metacells": len(meta),
                         "n_patients": meta.patient_id.nunique(), "beta": float(fit.params[term]),
                         "cluster_robust_se": float(fit.bse[term]), "p_value": float(fit.pvalues[term])})
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh(out.p_value)
    return out


def state_program_profiles(meta):
    rows = []
    for state in EMT_STATES:
        part = meta.loc[meta.emt_state.eq(state)]
        for program in PROGRAMS:
            col = "program::" + program
            patient_means = pd.Series({
                patient: weighted_mean(group[col], group.n_cells)
                for patient, group in part.groupby("patient_id")
            }, dtype=float)
            rows.append({"emt_state": state, "program": program, "metacells": len(part),
                         "patients_represented": int(part.patient_id.nunique()),
                         "mean_of_patient_weighted_means": float(patient_means.mean()),
                         "median_of_patient_weighted_means": float(patient_means.median())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", choices=["cpu"], default="cpu",
                        help="CPU-only by design; avoids anndata/PyTorch OpenMP collision")
    args = parser.parse_args()
    if args.permutations < 10000 or args.bootstrap < 2000:
        raise ValueError("Locked run requires 10000 permutations and 2000 bootstraps")
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, obs_names, var_names = read_h5ad_matrix(H5AD)
    meta = pd.read_csv(META, sep="\t", dtype={"metacell_id": str, "patient_id": str})
    if matrix.shape[0] != len(meta) or set(obs_names.astype(str)) != set(meta.metacell_id):
        raise ValueError("Metacell h5ad/metadata identity mismatch")
    order = obs_names.get_indexer(meta.metacell_id)
    if np.any(order < 0):
        raise ValueError("Metacell reorder failure")
    matrix = matrix[order]
    ordered_names = obs_names[order]
    meta, coverage = program_scores(matrix, ordered_names, var_names, meta)
    meta = assign_states(meta)
    patients = patient_features(meta)

    composition = pd.read_csv(COMPOSITION, sep="\t", dtype={"orig.ident": str})
    wide = composition.pivot(index="orig.ident", columns="celltype_major", values="fraction").fillna(0.0)
    for col in ["T-cells", "Myeloid", "CAFs", "Endothelial", "B-cells", "PVL"]:
        if col not in wide:
            wide[col] = 0.0
    wide["T_to_myeloid_logratio"] = np.log((wide["T-cells"] + 1e-4) / (wide["Myeloid"] + 1e-4))
    wide = wide.reset_index().rename(columns={"orig.ident": "patient_id"})
    patients = patients.merge(wide[["patient_id"] + COMPOSITION_FEATURES], on="patient_id",
                              how="left", validate="one_to_one")
    if patients[COMPOSITION_FEATURES].isna().any().any() or len(patients) != 20:
        raise ValueError("Patient composition linkage failure")

    composition_corr = patient_correlations(
        patients, PATIENT_EMT, COMPOSITION_FEATURES, args.permutations, args.bootstrap,
        args.n_jobs, SEED)
    program_corr = patient_correlations(
        patients, PATIENT_EMT, ["program::" + x for x in PROGRAMS], args.permutations,
        args.bootstrap, args.n_jobs, SEED + 1000)
    within = within_patient_models(meta)
    profiles = state_program_profiles(meta)

    coverage.to_csv(OUT / "tumor_intrinsic_program_gene_coverage.tsv", sep="\t", index=False)
    meta.to_csv(OUT / "malignant_metacell_emt_program_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    patients.to_csv(OUT / "patient_malignant_emt_and_composition.tsv", sep="\t", index=False)
    composition_corr.to_csv(OUT / "patient_emt_vs_microenvironment_correlations.tsv", sep="\t", index=False)
    program_corr.to_csv(OUT / "patient_emt_vs_tumor_intrinsic_program_correlations.tsv", sep="\t", index=False)
    within.to_csv(OUT / "within_patient_metacell_program_models.tsv", sep="\t", index=False)
    profiles.to_csv(OUT / "emt_state_tumor_intrinsic_program_profiles.tsv", sep="\t", index=False)

    result = {
        "status": "PASS", "analysis_type": "outcome-free GSE176078 cell-origin analysis",
        "patients": int(patients.patient_id.nunique()), "malignant_metacells": int(len(meta)),
        "malignant_cells_represented": int(meta.n_cells.sum()),
        "subtypes": {str(k): int(v) for k, v in patients.subtype.value_counts().items()},
        "state_metacells": {str(k): int(v) for k, v in meta.emt_state.value_counts().items()},
        "state_cells_represented": {str(k): int(v) for k, v in meta.groupby("emt_state").n_cells.sum().items()},
        "program_minimum_gene_coverage": float(coverage.fraction.min()),
        "patient_composition_tests": int(len(composition_corr)),
        "patient_composition_fdr_lt_0_05": int((composition_corr.permutation_fdr_bh < 0.05).sum()),
        "patient_program_tests": int(len(program_corr)),
        "patient_program_fdr_lt_0_05": int((program_corr.permutation_fdr_bh < 0.05).sum()),
        "within_patient_tests": int(len(within)),
        "within_patient_fdr_lt_0_05": int((within.fdr_bh < 0.05).sum()),
        "permutations": args.permutations, "bootstrap_replicates": args.bootstrap,
        "expression_program_backend": "CPU scipy sparse + NumPy",
        "h5ad_reader": "direct h5py; anndata/PyTorch not imported",
        "inference_backend": "CPU joblib + statsmodels (patient-level statistics)",
        "random_seed": SEED, "pseudoreplication_control": "patient primary; patient-clustered metacell secondary",
        "lock": str(LOCK), "lock_sha256": sha256(LOCK),
        "input_sha256": {str(p): sha256(p) for p in [H5AD, META, COMPOSITION]},
        "next_step": "spatial validation of malignant EMT states and immune proximity/exclusion",
    }
    (OUT / "gse176078_cell_origin_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
