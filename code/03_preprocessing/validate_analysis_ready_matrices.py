"""Independent integrity check of all five outcome-free analysis-ready HDF5 files."""
from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis/02_preprocessing/analysis_ready_validation_summary.json"
COMMON = pd.read_csv(ROOT / "02_data/02_interim/reference/cross_platform_gene_universe_v1/common_measured_gene_universe.tsv",
                     sep="\t")["gene_symbol"].astype(str).tolist()


def decode(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def inspect(cohort: str, path: Path, expected_ids: list[str], expected_genes: list[str]) -> dict:
    with h5py.File(path, "r") as h:
        samples, genes = decode(h["sample_id"]), decode(h["gene_symbol"])
        x = h["X"]
        nonfinite, mins, maxs = 0, [], []
        constant_min = np.full(len(genes), np.inf, dtype=np.float32)
        constant_max = np.full(len(genes), -np.inf, dtype=np.float32)
        for start in range(0, x.shape[0], 256):
            block = x[start:start + 256]
            nonfinite += int((~np.isfinite(block)).sum())
            mins.append(float(np.nanmin(block)))
            maxs.append(float(np.nanmax(block)))
            constant_min = np.minimum(constant_min, np.nanmin(block, axis=0))
            constant_max = np.maximum(constant_max, np.nanmax(block, axis=0))
        return {
            "cohort": cohort, "file": str(path), "sha256": sha(path),
            "shape": list(x.shape), "orientation": str(h.attrs.get("orientation", "")),
            "outcomes_included": bool(h.attrs.get("outcomes_included", True)),
            "sample_order_exact": samples == expected_ids,
            "gene_order_exact": genes == expected_genes,
            "duplicate_samples": len(samples) - len(set(samples)),
            "duplicate_genes": len(genes) - len(set(genes)),
            "nonfinite_values": nonfinite, "min": min(mins), "max": max(maxs),
            "constant_genes": int((constant_min == constant_max).sum()),
        }


def main() -> None:
    bman = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
    rman = ROOT / "02_data/02_interim/treatment_response/locked_manifests"
    probe = pd.read_csv(ROOT / "02_data/02_interim/treatment_response/probe_gene_lock_v1/locked_probe_to_gene_mapping.tsv",
                        sep="\t").sort_values("gene_symbol")
    response_genes = probe["gene_symbol"].astype(str).tolist()
    specs = [
        ("METABRIC", ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival/METABRIC_expression.h5",
         bman / "METABRIC_OS_locked_manifest.tsv", COMMON),
        ("TCGA_BRCA", ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival/TCGA_BRCA_expression.h5",
         bman / "TCGA_BRCA_OS_locked_manifest.tsv", COMMON),
        ("SCANB_GSE96058", ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival/SCANB_GSE96058_expression.h5",
         bman / "SCANB_GSE96058_OS_locked_manifest.tsv", COMMON),
        ("GSE25055", ROOT / "02_data/03_processed/analysis_ready_v1/treatment_response/GSE25055_expression.h5",
         rman / "GSE25066_discovery_pCR_manifest.tsv", response_genes),
        ("GSE25065", ROOT / "02_data/03_processed/analysis_ready_v1/treatment_response/GSE25065_expression.h5",
         rman / "GSE25066_validation_IDs_only_manifest.tsv", response_genes),
    ]
    results = []
    for cohort, path, manifest, genes in specs:
        ids = pd.read_csv(manifest, sep="\t")["sample_id"].astype(str).tolist()
        results.append(inspect(cohort, path, ids, genes))
    failures = [{"cohort": r["cohort"], "checks": [k for k, v in {
        "sample_order": r["sample_order_exact"], "gene_order": r["gene_order_exact"],
        "no_duplicate_samples": r["duplicate_samples"] == 0, "no_duplicate_genes": r["duplicate_genes"] == 0,
        "finite": r["nonfinite_values"] == 0, "outcome_free": not r["outcomes_included"],
        "orientation": r["orientation"] == "samples_x_genes"}.items() if not v]}
        for r in results if not (r["sample_order_exact"] and r["gene_order_exact"] and
                                 r["duplicate_samples"] == 0 and r["duplicate_genes"] == 0 and
                                 r["nonfinite_values"] == 0 and not r["outcomes_included"] and
                                 r["orientation"] == "samples_x_genes")]
    summary = {"status": "PASS" if not failures else "FAIL", "cohorts": results, "failures": failures,
               "next_gate": "candidate_rank_score_qc"}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failures": failures, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
