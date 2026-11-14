#!/usr/bin/env python
"""Build outcome-free patient-paired malignant EMT-state discovery signatures."""
from __future__ import annotations

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
from scipy import sparse
from scipy.stats import ttest_1samp


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
H5AD = ROOT / "02_data/03_processed/single_cell/GSE176078/GSE176078_malignant_patient_metacells.h5ad"
META = ROOT / "04_analysis/08_biological_landscape/single_cell/GSE176078_cell_origin/malignant_metacell_emt_program_scores.tsv.gz"
LINCS_GENES = ROOT / "02_data/01_raw/reference/lincs/GSE70138/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/01_state_signature_discovery/GSE176078_malignant"

STATES = ["epithelial_dominant", "hybrid_EM", "mesenchymal_dominant"]
MIN_PAIRED_PATIENTS = 6
MIN_MEAN_CPM = 1.0
MIN_ABS_LOG2FC = 0.25
MIN_CONSISTENCY = 2 / 3
MAX_GENES_PER_DIRECTION = 100
MIN_GENES_PER_DIRECTION = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        obs = decode(handle["obs"][obs_key][...])
        genes = decode(handle["var"][var_key][...]).str.upper()
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            matrix = sparse.csr_matrix(node[...])
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            shape = tuple(int(x) for x in node.attrs["shape"])
            data = node["data"][...]
            indices = node["indices"][...]
            indptr = node["indptr"][...]
            if encoding == "csr_matrix":
                matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
            elif encoding == "csc_matrix":
                matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Unsupported X encoding: {encoding}")
    return matrix.tocsr(), obs.astype(str), genes.astype(str)


def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    result = np.ones(len(p), float)
    good = np.isfinite(p)
    if not good.any():
        return result
    valid = p[good]
    order = np.argsort(valid)
    adjusted = valid[order] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[good] = restored
    return result


def aggregate_rows(matrix: sparse.csr_matrix, indices: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices].sum(axis=0)).ravel().astype(float)


def log2cpm(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    libraries = counts.sum(axis=1)
    if np.any(libraries <= 0):
        raise ValueError("Zero pseudobulk library")
    cpm = counts / libraries[:, None] * 1e6
    return np.log2(cpm + 0.5), cpm


def main() -> None:
    for path in [H5AD, META, LINCS_GENES, LOCK]:
        if not path.exists():
            raise FileNotFoundError(path)

    matrix, obs_names, genes = read_h5ad(H5AD)
    if matrix.shape != (len(obs_names), len(genes)):
        raise ValueError("H5AD dimension mismatch")
    if genes.duplicated().any():
        raise ValueError("Duplicated upper-case gene symbols")
    if matrix.data.size and (not np.isfinite(matrix.data).all() or matrix.data.min() < 0):
        raise ValueError("Expected finite non-negative metacell counts")

    metadata = pd.read_csv(META, sep="\t", compression="gzip")
    metadata["metacell_id"] = metadata["metacell_id"].astype(str)
    metadata = metadata.set_index("metacell_id").loc[obs_names].reset_index()
    if metadata["patient_id"].isna().any() or metadata["emt_state"].isna().any():
        raise ValueError("Missing patient/state metadata")

    lincs = pd.read_csv(LINCS_GENES, sep="\t", compression="gzip", dtype=str)
    lincs["pr_gene_symbol"] = lincs["pr_gene_symbol"].astype(str).str.upper()
    lincs_bing = set(lincs.loc[pd.to_numeric(lincs["pr_is_bing"], errors="coerce").eq(1), "pr_gene_symbol"])
    lincs_landmark = set(lincs.loc[pd.to_numeric(lincs["pr_is_lm"], errors="coerce").eq(1), "pr_gene_symbol"])

    all_stats = []
    pair_manifest = []
    pseudobulk_rows = []
    failures = []

    for state in STATES:
        comparator_states = [value for value in STATES if value != state]
        target_counts, comparator_counts, paired_patients = [], [], []
        for patient, group in metadata.groupby("patient_id", sort=True):
            target_idx = group.index[group["emt_state"].eq(state)].to_numpy()
            comparator_idx = group.index[group["emt_state"].isin(comparator_states)].to_numpy()
            if not len(target_idx) or not len(comparator_idx):
                continue
            target = aggregate_rows(matrix, target_idx)
            comparator = aggregate_rows(matrix, comparator_idx)
            target_counts.append(target)
            comparator_counts.append(comparator)
            paired_patients.append(str(patient))
            pair_manifest.append({
                "state": state,
                "patient_id": str(patient),
                "target_metacells": int(len(target_idx)),
                "comparator_metacells": int(len(comparator_idx)),
                "target_cells_represented": int(group.loc[target_idx, "n_cells"].astype(int).sum()),
                "comparator_cells_represented": int(group.loc[comparator_idx, "n_cells"].astype(int).sum()),
            })

        if len(paired_patients) < MIN_PAIRED_PATIENTS:
            failures.append(f"insufficient_paired_patients:{state}:{len(paired_patients)}")
            continue
        target_counts = np.vstack(target_counts)
        comparator_counts = np.vstack(comparator_counts)
        target_log, target_cpm = log2cpm(target_counts)
        comparator_log, comparator_cpm = log2cpm(comparator_counts)
        differences = target_log - comparator_log

        test = ttest_1samp(differences, popmean=0.0, axis=0, nan_policy="omit")
        t_stat = np.nan_to_num(np.asarray(test.statistic, float), nan=0.0, posinf=0.0, neginf=0.0)
        p_value = np.nan_to_num(np.asarray(test.pvalue, float), nan=1.0, posinf=1.0, neginf=1.0)
        mean_effect = differences.mean(axis=0)
        median_effect = np.median(differences, axis=0)
        positive_fraction = (differences > 0).mean(axis=0)
        negative_fraction = (differences < 0).mean(axis=0)
        consistency = np.maximum(positive_fraction, negative_fraction)
        mean_cpm = np.vstack([target_cpm, comparator_cpm]).mean(axis=0)

        stats = pd.DataFrame({
            "state": state,
            "gene": genes,
            "paired_patients": len(paired_patients),
            "mean_paired_log2fc": mean_effect,
            "median_paired_log2fc": median_effect,
            "paired_t": t_stat,
            "p_value": p_value,
            "fdr_bh": bh(p_value),
            "positive_patient_fraction": positive_fraction,
            "negative_patient_fraction": negative_fraction,
            "direction_consistency": consistency,
            "mean_cpm": mean_cpm,
            "lincs_bing": [gene in lincs_bing for gene in genes],
            "lincs_landmark": [gene in lincs_landmark for gene in genes],
        })
        stats["eligible_for_query"] = (
            stats.mean_cpm.ge(MIN_MEAN_CPM)
            & stats.mean_paired_log2fc.abs().ge(MIN_ABS_LOG2FC)
            & stats.direction_consistency.ge(MIN_CONSISTENCY)
            & stats.lincs_bing
        )
        all_stats.append(stats)

        for patient_index, patient in enumerate(paired_patients):
            pseudobulk_rows.extend([
                {"state": state, "patient_id": patient, "group": "target", "library_size": int(target_counts[patient_index].sum())},
                {"state": state, "patient_id": patient, "group": "comparator", "library_size": int(comparator_counts[patient_index].sum())},
            ])

    if not all_stats:
        raise ValueError(f"No state signature could be estimated: {failures}")
    statistics = pd.concat(all_stats, ignore_index=True)

    query_rows = []
    for state in STATES:
        state_stats = statistics.loc[(statistics.state == state) & statistics.eligible_for_query].copy()
        for direction, sign in [("up", 1), ("down", -1)]:
            candidates = state_stats.loc[np.sign(state_stats.mean_paired_log2fc) == sign].copy()
            candidates = candidates.sort_values(
                ["paired_t", "mean_paired_log2fc"],
                ascending=[direction == "down", direction == "down"],
            ).head(MAX_GENES_PER_DIRECTION)
            if len(candidates) < MIN_GENES_PER_DIRECTION:
                failures.append(f"insufficient_query_genes:{state}:{direction}:{len(candidates)}")
            for rank, row in enumerate(candidates.itertuples(), start=1):
                query_rows.append({
                    "state": state,
                    "disease_direction": direction,
                    "rank": rank,
                    "gene": row.gene,
                    "mean_paired_log2fc": row.mean_paired_log2fc,
                    "direction_consistency": row.direction_consistency,
                    "paired_t": row.paired_t,
                    "p_value": row.p_value,
                    "fdr_bh": row.fdr_bh,
                    "lincs_landmark": row.lincs_landmark,
                })
    queries = pd.DataFrame(query_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(OUT / "malignant_state_gene_statistics.tsv.gz", sep="\t", index=False, compression="gzip")
    queries.to_csv(OUT / "lincs_candidate_state_query_genes.tsv", sep="\t", index=False)
    pd.DataFrame(pair_manifest).to_csv(OUT / "paired_patient_state_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(pseudobulk_rows).to_csv(OUT / "pseudobulk_library_manifest.tsv", sep="\t", index=False)

    counts = queries.groupby(["state", "disease_direction"]).size().unstack(fill_value=0) if len(queries) else pd.DataFrame()
    summary = {
        "status": "FAIL" if failures else "PASS",
        "analysis_type": "outcome-free patient-paired malignant EMT-state discovery signatures",
        "outcomes_used": False,
        "treatment_response_used": False,
        "metacells": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "genes_tested_per_state": int(len(genes)),
        "paired_patients": {
            state: int(pd.DataFrame(pair_manifest).query("state == @state").patient_id.nunique())
            for state in STATES
        },
        "query_gene_counts": {
            state: {direction: int(counts.loc[state, direction]) if state in counts.index and direction in counts.columns else 0
                    for direction in ["up", "down"]}
            for state in STATES
        },
        "query_definition": {
            "minimum_mean_cpm": MIN_MEAN_CPM,
            "minimum_absolute_mean_paired_log2fc": MIN_ABS_LOG2FC,
            "minimum_direction_consistency": MIN_CONSISTENCY,
            "lincs_gene_space": "best inferred (BING)",
            "maximum_genes_per_direction": MAX_GENES_PER_DIRECTION,
        },
        "failures": failures,
        "input_sha256": {str(path): sha256(path) for path in [H5AD, META, LINCS_GENES, LOCK]},
        "next_gate": "independent TCGA/METABRIC/SCAN-B directional replication before compound connectivity scoring",
    }
    (OUT / "malignant_state_signature_discovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
