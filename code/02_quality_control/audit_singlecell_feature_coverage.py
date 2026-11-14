"""Read only H5AD gene identifiers and candidate signature coverage; no expression values."""
from __future__ import annotations

import os

import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis/01_data_audit/singlecell_spatial/feature_coverage_summary.json"
EMP = ROOT / "02_data/02_interim/externally_anchored_emp_v5.0_locked/signatures/locked_external_emp_signatures.tsv"
TIP = ROOT / "02_data/02_interim/reference/tip_cancer_immunity_cycle/TIP_cancer_immunity_cycle_28_signatures.gmt"
GTF = ROOT / "02_data/01_raw/reference/gencode/gencode.v50.annotation.gtf.gz"
FILES = {
    "Integrated_BRCA_Atlas_2026": ROOT / "02_data/01_raw/single_cell/Integrated_BRCA_Atlas_2026/Integrated_BRCA_Global_Atlas_2026.h5ad",
    "GSE176078": ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad",
    "GSE225600_sc": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_sc_raw_counts.h5ad",
    "GSE225600_ST": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_ST_raw_counts.h5ad",
}


def decode(a) -> list[str]:
    arr = a[:]
    return [x.decode("utf-8", "replace") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]


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


def genes(path: Path, reference: set[str], emap: dict[str, str]) -> set[str]:
    with h5py.File(path, "r") as h:
        var = h["var"]
        idx = var.attrs.get("_index", "_index")
        if isinstance(idx, bytes):
            idx = idx.decode()
        candidates = [{x.upper() for x in decode(var[idx])}]
        for candidate in ("gene_name", "gene_symbol", "feature_name", "symbol"):
            if candidate in var and isinstance(var[candidate], h5py.Dataset):
                candidates.append({x.upper() for x in decode(var[candidate])})
        expanded = []
        for candidate in candidates:
            expanded.append(candidate)
            mapped = {emap.get(x.split(".")[0], "") for x in candidate if x.startswith("ENSG")}
            mapped.discard("")
            if mapped:
                expanded.append(mapped)
        return max(expanded, key=lambda x: len(x & reference))


def sigs() -> dict[str, set[str]]:
    e = pd.read_csv(EMP, sep="\t")
    out = {f"candidate_EMP::{k}": set(v.gene.astype(str).str.upper()) for k, v in e.groupby("signature")}
    with TIP.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip().split("\t")
            if len(p) > 2:
                out[f"TIP::{p[0]}"] = {x.upper() for x in p[2:] if x}
    return out


def main() -> None:
    signatures, assets, failures = sigs(), [], []
    reference = set().union(*signatures.values())
    emap = ensembl_map()
    for name, path in FILES.items():
        universe = genes(path, reference, emap)
        rows = []
        for sname, sgenes in signatures.items():
            n = len(sgenes & universe)
            rows.append({"signature": sname, "n_total": len(sgenes), "n_present": n,
                         "coverage": round(n / len(sgenes), 4), "passes_80pct": n / len(sgenes) >= .8})
        assets.append({"asset": name, "unique_features": len(universe), "coverage": rows})
        failures.extend({"asset": name, **r} for r in rows if not r["passes_80pct"])
    summary = {"status": "PASS", "expression_values_opened": False, "assets": assets,
               "coverage_failures_below_80pct": failures,
               "next_gate": "select_single_cell_primary_and_replication_roles"}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "assets": len(assets), "coverage_failures": len(failures),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
