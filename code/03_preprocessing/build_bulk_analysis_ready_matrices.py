"""Build score-free, outcome-free HDF5 expression matrices for the survival branch."""
from __future__ import annotations

import os

import gzip
import hashlib
import json
import re
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
COMMON_FILE = ROOT / "02_data/02_interim/reference/cross_platform_gene_universe_v1/common_measured_gene_universe.tsv"
MANIFESTS = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
DEST = ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival"
REPORT = ROOT / "04_analysis/02_preprocessing/bulk_analysis_ready_summary.json"
GTF = ROOT / "02_data/01_raw/reference/gencode/gencode.v50.annotation.gtf.gz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ensembl_map() -> dict[str, str]:
    out = {}
    rid, rname = re.compile(r'gene_id "([^"]+)"'), re.compile(r'gene_name "([^"]+)"')
    with gzip.open(GTF, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            a, b = rid.search(line), rname.search(line)
            if a and b:
                out[a.group(1).split(".")[0]] = b.group(1).upper()
    return out


def write_h5(path: Path, matrix: pd.DataFrame, cohort: str) -> dict:
    x = matrix.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError(f"{cohort}: non-finite expression values")
    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h.create_dataset("X", data=x, compression="gzip", compression_opts=4,
                         chunks=(min(128, x.shape[0]), min(512, x.shape[1])))
        h.create_dataset("sample_id", data=matrix.index.astype(str).to_numpy(), dtype=text)
        h.create_dataset("gene_symbol", data=matrix.columns.astype(str).to_numpy(), dtype=text)
        h.attrs["orientation"] = "samples_x_genes"
        h.attrs["cohort"] = cohort
        h.attrs["outcomes_included"] = False
    return {"cohort": cohort, "samples": x.shape[0], "genes": x.shape[1],
            "min": float(x.min()), "max": float(x.max()), "constant_genes": int((x.std(axis=0) == 0).sum()),
            "file": str(path), "sha256": sha(path)}


def build_matrix(path: Path, sep: str, id_columns: list[str], sample_map: dict[str, str],
                 common: list[str], mapper) -> tuple[pd.DataFrame, dict]:
    common_set = set(common)
    selected_source_samples = list(sample_map)
    usecols = [*id_columns, *selected_source_samples]
    pieces = []
    for chunk in pd.read_csv(path, sep=sep, usecols=usecols, chunksize=500, low_memory=False):
        symbols = chunk[id_columns[0]].astype(str).map(mapper)
        keep = symbols.isin(common_set)
        if not keep.any():
            continue
        values = chunk.loc[keep, selected_source_samples].apply(pd.to_numeric, errors="coerce")
        values.index = symbols[keep].to_numpy()
        pieces.append(values)
    if not pieces:
        raise RuntimeError(f"No common genes read from {path}")
    genes_by_samples = pd.concat(pieces).groupby(level=0, sort=False).median()
    genes_by_samples = genes_by_samples.reindex(common)
    missing = genes_by_samples.isna()
    missing_cells = int(missing.sum().sum())
    missing_rate = missing_cells / int(genes_by_samples.size)
    fully_missing_genes = genes_by_samples.index[missing.all(axis=1)].tolist()
    if fully_missing_genes:
        raise ValueError(f"Fully missing common genes: {fully_missing_genes[:20]}")
    if missing_rate > 0.005:
        raise ValueError(f"Expression missingness exceeds 0.5%: cells={missing_cells}, rate={missing_rate:.6f}")
    samples_with_missing = int(missing.any(axis=0).sum())
    genes_with_missing = int(missing.any(axis=1).sum())
    if missing_cells:
        sample_medians = genes_by_samples.median(axis=0, skipna=True)
        genes_by_samples = genes_by_samples.fillna(sample_medians)
    if genes_by_samples.isna().any().any():
        raise ValueError("Non-finite values remain after within-sample median imputation")
    genes_by_samples = genes_by_samples.rename(columns=sample_map)
    qc = {"missing_cells_before_imputation": missing_cells,
          "missing_rate_before_imputation": missing_rate,
          "samples_with_missing_before_imputation": samples_with_missing,
          "genes_with_missing_before_imputation": genes_with_missing,
          "imputation": "within_sample_median" if missing_cells else "none"}
    return genes_by_samples.T, qc


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    failure_file = REPORT.parent / "bulk_analysis_ready_failure.json"
    if failure_file.exists():
        failure_file.unlink()
    common = pd.read_csv(COMMON_FILE, sep="\t")["gene_symbol"].astype(str).str.upper().tolist()
    emap = ensembl_map()
    results = []

    m_ids = pd.read_csv(MANIFESTS / "METABRIC_OS_locked_manifest.tsv", sep="\t")["sample_id"].astype(str).tolist()
    m, m_qc = build_matrix(ROOT / "02_data/01_raw/external_validation/metabric/brca_metabric/data_mrna_illumina_microarray.txt",
                           "\t", ["Hugo_Symbol"], {x: x for x in m_ids}, common, lambda x: x.upper())
    m_result = write_h5(DEST / "METABRIC_expression.h5", m.loc[m_ids, common], "METABRIC")
    m_result["missing_value_qc"] = m_qc
    results.append(m_result)

    t_ids = pd.read_csv(MANIFESTS / "TCGA_BRCA_OS_locked_manifest.tsv", sep="\t")["sample_id"].astype(str).tolist()
    t_path = ROOT / "02_data/01_raw/bulk_rnaseq/tcga_brca_xena/expression/TCGA-BRCA.star_counts.tsv.gz"
    t_header = pd.read_csv(t_path, sep="\t", nrows=0).columns.astype(str).tolist()
    candidates = {}
    for col in t_header[1:]:
        if col.startswith("TCGA-") and len(col) >= 15 and col[13:15] == "01":
            candidates.setdefault(col[:12], []).append(col)
    t_map = {sorted(candidates[p])[0]: p for p in t_ids if p in candidates}
    if len(t_map) != len(t_ids):
        raise ValueError(f"TCGA sample mapping incomplete: {len(t_map)}/{len(t_ids)}")
    t, t_qc = build_matrix(t_path, "\t", ["Ensembl_ID"], t_map, common,
                           lambda x: emap.get(x.split(".")[0], ""))
    t_result = write_h5(DEST / "TCGA_BRCA_expression.h5", t.loc[t_ids, common], "TCGA_BRCA")
    t_result["missing_value_qc"] = t_qc
    results.append(t_result)

    s_ids = pd.read_csv(MANIFESTS / "SCANB_GSE96058_OS_locked_manifest.tsv", sep="\t")["sample_id"].astype(str).tolist()
    score_path = ROOT / "02_data/03_processed/bulk_validation/SCANB_GSE96058_scores.tsv.gz"
    link = pd.read_csv(score_path, sep="\t", usecols=[0, 1], low_memory=False)
    fcol, gcol = link.columns[0], link.columns[1]
    lookup = dict(zip(link[gcol].astype(str), link[fcol].astype(str)))
    s_map = {lookup[g]: g for g in s_ids if g in lookup}
    if len(s_map) != len(s_ids):
        raise ValueError(f"SCAN-B sample mapping incomplete: {len(s_map)}/{len(s_ids)}")
    s, s_qc = build_matrix(ROOT / "02_data/01_raw/external_validation/GSE96058/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz",
                           ",", ["Unnamed: 0"], s_map, common, lambda x: x.upper())
    s_result = write_h5(DEST / "SCANB_GSE96058_expression.h5", s.loc[s_ids, common], "SCANB_GSE96058")
    s_result["missing_value_qc"] = s_qc
    results.append(s_result)

    summary = {"status": "PASS", "version": "1.0", "outcomes_included": False,
               "common_gene_count": len(common), "cohorts": results,
               "next_gate": "analysis_ready_matrix_validation_and_rank_score_construction"}
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        (REPORT.parent / "bulk_analysis_ready_failure.json").write_text(
            json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
