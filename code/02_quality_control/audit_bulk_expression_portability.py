"""Outcome-blind audit of raw bulk expression identifiers and signature coverage."""
from __future__ import annotations

import os

import gzip
import json
import re
import csv
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis/01_data_audit/expression_portability"
EMP = ROOT / "02_data/02_interim/externally_anchored_emp_v5.0_locked/signatures/locked_external_emp_signatures.tsv"
TIP = ROOT / "02_data/02_interim/reference/tip_cancer_immunity_cycle/TIP_cancer_immunity_cycle_28_signatures.gmt"
GTF = ROOT / "02_data/01_raw/reference/gencode/gencode.v50.annotation.gtf.gz"


def signatures() -> dict[str, set[str]]:
    e = pd.read_csv(EMP, sep="\t")
    sig = {f"candidate_EMP::{k}": set(v["gene"].astype(str).str.upper()) for k, v in e.groupby("signature")}
    with TIP.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2:
                sig[f"TIP::{p[0]}"] = {x.upper() for x in p[2:] if x}
    return sig


def ensembl_map() -> dict[str, str]:
    mapping = {}
    rx_id, rx_name = re.compile(r'gene_id "([^"]+)"'), re.compile(r'gene_name "([^"]+)"')
    with gzip.open(GTF, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            a, b = rx_id.search(line), rx_name.search(line)
            if a and b:
                mapping[a.group(1).split(".")[0]] = b.group(1).upper()
    return mapping


def coverage(universe: set[str], sigs: dict[str, set[str]], cohort: str) -> list[dict]:
    return [{"cohort": cohort, "signature": name, "n_total": len(genes),
             "n_present": len(genes & universe), "coverage": round(len(genes & universe) / len(genes), 4),
             "passes_80pct": len(genes & universe) / len(genes) >= 0.80}
            for name, genes in sigs.items() if genes]


def matrix_axes(path: Path, delimiter: str) -> tuple[list[str], list[str]]:
    """Stream a wide matrix so only its header and first field enter memory."""
    opener = gzip.open if path.suffix == ".gz" else open
    row_ids = []
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as f:
        header = next(csv.reader([f.readline()], delimiter=delimiter))
        for line in f:
            if line.strip():
                row_ids.append(line.split(delimiter, 1)[0].strip().strip('"'))
    return header, row_ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failure_file = OUT / "bulk_expression_portability_failure.json"
    if failure_file.exists():
        failure_file.unlink()
    sigs, emap = signatures(), ensembl_map()
    specs = {
        "METABRIC": ROOT / "02_data/01_raw/external_validation/metabric/brca_metabric/data_mrna_illumina_microarray.txt",
        "TCGA_BRCA": ROOT / "02_data/01_raw/bulk_rnaseq/tcga_brca_xena/expression/TCGA-BRCA.star_counts.tsv.gz",
        "SCANB_GSE96058": ROOT / "02_data/01_raw/external_validation/GSE96058/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz",
    }
    manifests = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
    results, cov = [], []
    for cohort, path in specs.items():
        sep = "," if cohort.startswith("SCANB") else "\t"
        header, raw_ids = matrix_axes(path, sep)
        ids = pd.Series(raw_ids, dtype="string")
        if cohort == "TCGA_BRCA":
            universe = {emap.get(x.split(".")[0], "") for x in ids}
            expr_samples = {x[:12] for x in header[1:] if x.startswith("TCGA-") and "-01" in x}
        elif cohort == "METABRIC":
            universe = set(ids.str.upper())
            expr_samples = set(header[2:])
        else:
            universe = set(ids.str.upper())
            mapdf = pd.read_csv(ROOT / "02_data/03_processed/bulk_validation/SCANB_GSE96058_scores.tsv.gz",
                                sep="\t", usecols=[0, 1], low_memory=False)
            valid_cols = set(header[1:])
            matrix_key, geo_key = mapdf.columns[0], mapdf.columns[1]
            expr_samples = set(mapdf.loc[mapdf[matrix_key].astype(str).isin(valid_cols), geo_key].astype(str))
        universe.discard("")
        mpath = manifests / f"{cohort}_OS_locked_manifest.tsv"
        locked = set(pd.read_csv(mpath, sep="\t", usecols=["sample_id"])["sample_id"].astype(str))
        overlap = locked & expr_samples
        results.append({"cohort": cohort, "matrix_rows": int(len(ids)), "matrix_columns": len(header),
                        "unique_gene_symbols": len(universe), "expression_sample_ids": len(expr_samples),
                        "locked_samples": len(locked), "locked_with_expression": len(overlap),
                        "locked_expression_coverage": round(len(overlap) / len(locked), 4),
                        "missing_locked_sample_ids": sorted(locked - expr_samples)})
        cov.extend(coverage(universe, sigs, cohort))
        pd.Series(sorted(universe), name="gene_symbol").to_csv(OUT / f"{cohort}_gene_universe.tsv", sep="\t", index=False)
    pd.DataFrame(cov).to_csv(OUT / "bulk_signature_coverage.tsv", sep="\t", index=False)
    status = "PASS" if all(x["locked_expression_coverage"] >= .98 for x in results) else "REVIEW"
    summary = {"status": status, "outcome_used": False, "cohorts": results,
               "signature_sets_are_candidates_not_final_model_features": True,
               "coverage_failures_below_80pct": [x for x in cov if not x["passes_80pct"]],
               "next_gate": "cross_platform_gene_universe_and_preprocessing_specification"}
    (OUT / "bulk_expression_portability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "cohorts": results, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "bulk_expression_portability_failure.json").write_text(
            json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
