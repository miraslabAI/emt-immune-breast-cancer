#!/usr/bin/env python
"""Pathway-first malignant EMT-state analysis using Hallmark and Reactome."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as t_distribution
from scipy.stats import ttest_1samp


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
H5AD = ROOT / "02_data/03_processed/single_cell/GSE176078/GSE176078_malignant_patient_metacells.h5ad"
META = ROOT / "04_analysis/08_biological_landscape/single_cell/GSE176078_cell_origin/malignant_metacell_emt_program_scores.tsv.gz"
HALLMARK = ROOT / "02_data/01_raw/reference/msigdb/h.all.v2026.1.Hs.symbols.gmt"
REACTOME = ROOT / "02_data/01_raw/reference/reactome/ReactomePathways.gmt.zip"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/03_state_pathway_analysis/GSE176078_malignant"

STATES = ["epithelial_dominant", "hybrid_EM", "mesenchymal_dominant"]
CONTRASTS = [
    ("hybrid_minus_epithelial", "hybrid_EM", "epithelial_dominant"),
    ("mesenchymal_minus_hybrid", "mesenchymal_dominant", "hybrid_EM"),
    ("mesenchymal_minus_epithelial", "mesenchymal_dominant", "epithelial_dominant"),
]
MIN_GENES = 10
MAX_GENES = 500


def decode(values) -> pd.Index:
    return pd.Index([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ])


def read_h5ad(path: Path):
    with h5py.File(path, "r") as handle:
        obs_key = handle["obs"].attrs.get("_index", "_index")
        var_key = handle["var"].attrs.get("_index", "_index")
        if isinstance(obs_key, bytes):
            obs_key = obs_key.decode()
        if isinstance(var_key, bytes):
            var_key = var_key.decode()
        obs = decode(handle["obs"][obs_key][...]).astype(str)
        genes = decode(handle["var"][var_key][...]).str.upper().astype(str)
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            matrix = sparse.csr_matrix(node[...])
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            shape = tuple(int(v) for v in node.attrs["shape"])
            data, indices, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
            if encoding == "csr_matrix":
                matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
            elif encoding == "csc_matrix":
                matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Unsupported X encoding: {encoding}")
    return matrix.tocsr(), obs, genes


def parse_gmt(lines, database: str) -> list[dict]:
    rows = []
    for raw in lines:
        parts = raw.rstrip("\n\r").split("\t")
        if len(parts) >= 3:
            rows.append({"database": database, "pathway": parts[0], "genes": {g.upper() for g in parts[2:] if g}})
    return rows


def load_pathways() -> list[dict]:
    pathways = parse_gmt(HALLMARK.read_text(encoding="utf-8").splitlines(), "Hallmark")
    with zipfile.ZipFile(REACTOME) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Unexpected Reactome members: {names}")
        with archive.open(names[0]) as handle:
            pathways.extend(parse_gmt((line.decode("utf-8") for line in handle), "Reactome"))
    return pathways


def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    output = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return output
    x = p[valid]
    order = np.argsort(x)
    adjusted = x[order] * len(x) / np.arange(1, len(x) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    output[valid] = restored
    return output


def cluster_robust_ols(y: np.ndarray, metadata: pd.DataFrame):
    patient_dummies = pd.get_dummies(metadata.patient_id.astype(str), drop_first=True, dtype=float)
    x_frame = pd.concat([
        pd.Series(1.0, index=metadata.index, name="intercept"),
        metadata[["emt_position_within_patient", "coactivation_within_patient", "cell_cycle_rank", "stress_rank"]].astype(float),
        patient_dummies,
    ], axis=1)
    x = x_frame.to_numpy(float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    groups = metadata.patient_id.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n, k = x.shape
    correction = (len(unique_groups) / (len(unique_groups) - 1)) * ((n - 1) / (n - k))
    rows = []
    for term in ["emt_position_within_patient", "coactivation_within_patient"]:
        j = list(x_frame.columns).index(term)
        variance = np.zeros(y.shape[1], float)
        for group in unique_groups:
            idx = np.flatnonzero(groups == group)
            score = x[idx].T @ residual[idx]
            projected = bread[j] @ score
            variance += projected ** 2
        se = np.sqrt(np.maximum(variance * correction, 0))
        t_value = np.divide(beta[j], se, out=np.zeros_like(se), where=se > 0)
        p_value = 2 * t_distribution.sf(np.abs(t_value), df=len(unique_groups) - 1)
        rows.append((term, beta[j], se, t_value, p_value))
    return rows


def main() -> None:
    for path in [H5AD, META, HALLMARK, REACTOME, LOCK]:
        if not path.exists():
            raise FileNotFoundError(path)
    matrix, obs_names, genes = read_h5ad(H5AD)
    metadata = pd.read_csv(META, sep="\t", compression="gzip")
    metadata.metacell_id = metadata.metacell_id.astype(str)
    metadata = metadata.set_index("metacell_id").loc[obs_names].reset_index()
    if genes.duplicated().any():
        raise ValueError("Duplicated upper-case gene symbols")

    library = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("Zero metacell library")
    normalized = sparse.diags(1e6 / library) @ matrix
    normalized.data = np.log1p(normalized.data)
    values = normalized.toarray().astype(np.float32)
    means = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    variable = sd > 0
    values[:, variable] = (values[:, variable] - means[variable]) / sd[variable]
    values[:, ~variable] = 0

    gene_lookup = {gene: i for i, gene in enumerate(genes)}
    pathway_rows, membership_rows, membership_cols, membership_data = [], [], [], []
    for candidate in load_pathways():
        detected = sorted(candidate["genes"] & set(gene_lookup))
        if not (MIN_GENES <= len(detected) <= MAX_GENES):
            continue
        column = len(pathway_rows)
        pathway_rows.append({
            "database": candidate["database"], "pathway": candidate["pathway"],
            "source_genes": len(candidate["genes"]), "detected_genes": len(detected),
            "detected_gene_symbols": ";".join(detected),
        })
        weight = 1.0 / len(detected)
        for gene in detected:
            membership_rows.append(gene_lookup[gene])
            membership_cols.append(column)
            membership_data.append(weight)
    pathway_manifest = pd.DataFrame(pathway_rows)
    membership = sparse.csc_matrix(
        (membership_data, (membership_rows, membership_cols)),
        shape=(len(genes), len(pathway_manifest)), dtype=np.float32,
    )
    scores = np.asarray(values @ membership, dtype=np.float32)

    continuous_rows = []
    for term, beta, se, t_value, p_value in cluster_robust_ols(scores, metadata):
        block = pathway_manifest[["database", "pathway", "detected_genes"]].copy()
        block["term"] = term
        block["n_metacells"] = len(metadata)
        block["n_patient_clusters"] = metadata.patient_id.nunique()
        block["beta"] = beta
        block["cluster_robust_se"] = se
        block["t_value"] = t_value
        block["p_value"] = p_value
        continuous_rows.append(block)
    continuous = pd.concat(continuous_rows, ignore_index=True)
    continuous["fdr_bh"] = continuous.groupby(["database", "term"], group_keys=False).p_value.transform(bh)

    state_rows = []
    weights = pd.to_numeric(metadata.n_cells, errors="raise").to_numpy(float)
    for (patient, state), idx in metadata.groupby(["patient_id", "emt_state"]).groups.items():
        idx = np.asarray(list(idx), int)
        weighted = np.average(scores[idx], axis=0, weights=weights[idx])
        for column, value in enumerate(weighted):
            state_rows.append({
                "patient_id": patient, "emt_state": state,
                "database": pathway_manifest.iloc[column].database,
                "pathway": pathway_manifest.iloc[column].pathway,
                "pathway_score": float(value),
            })
    patient_state = pd.DataFrame(state_rows)

    contrast_rows = []
    for contrast, numerator, denominator in CONTRASTS:
        left = patient_state.loc[patient_state.emt_state.eq(numerator)]
        right = patient_state.loc[patient_state.emt_state.eq(denominator)]
        paired = left.merge(right, on=["patient_id", "database", "pathway"], suffixes=("_num", "_den"))
        for (database, pathway), group in paired.groupby(["database", "pathway"], sort=False):
            difference = group.pathway_score_num.to_numpy() - group.pathway_score_den.to_numpy()
            test = ttest_1samp(difference, 0.0)
            contrast_rows.append({
                "contrast": contrast, "numerator": numerator, "denominator": denominator,
                "database": database, "pathway": pathway, "paired_patients": len(difference),
                "mean_paired_difference": float(np.mean(difference)),
                "median_paired_difference": float(np.median(difference)),
                "paired_t": float(test.statistic) if np.isfinite(test.statistic) else 0.0,
                "p_value": float(test.pvalue) if np.isfinite(test.pvalue) else 1.0,
                "low_information_lt6_pairs": len(difference) < 6,
            })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["fdr_bh"] = contrasts.groupby(["database", "contrast"], group_keys=False).p_value.transform(bh)

    profile = patient_state.groupby(["database", "pathway", "emt_state"]).agg(
        patients=("patient_id", "nunique"), mean_score=("pathway_score", "mean"),
        median_score=("pathway_score", "median")
    ).reset_index()

    OUT.mkdir(parents=True, exist_ok=True)
    pathway_manifest.to_csv(OUT / "pathway_gene_coverage.tsv.gz", sep="\t", index=False, compression="gzip")
    continuous.to_csv(OUT / "continuous_emt_pathway_models.tsv.gz", sep="\t", index=False, compression="gzip")
    contrasts.to_csv(OUT / "paired_state_pathway_contrasts.tsv.gz", sep="\t", index=False, compression="gzip")
    profile.to_csv(OUT / "patient_weighted_state_pathway_profiles.tsv.gz", sep="\t", index=False, compression="gzip")
    summary = {
        "status": "PASS",
        "analysis_type": "outcome-free state-specific pathway analysis before drug ranking",
        "metacells": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "pathways_scored": {str(k): int(v) for k, v in pathway_manifest.database.value_counts().items()},
        "continuous_tests": int(len(continuous)),
        "continuous_fdr_lt_0_05": int((continuous.fdr_bh < 0.05).sum()),
        "paired_contrasts": {
            name: int(group.paired_patients.max()) for name, group in contrasts.groupby("contrast")
        },
        "paired_tests_fdr_lt_0_05": {
            name: int((group.fdr_bh < 0.05).sum()) for name, group in contrasts.groupby("contrast")
        },
        "low_information_rule": "paired contrasts with <6 patients cannot define a drug query alone",
        "next_gate": "collapse pathway redundancy, replicate state pathway directions in bulk cohorts, then build state-specific drug objectives",
    }
    (OUT / "gse176078_state_pathway_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
