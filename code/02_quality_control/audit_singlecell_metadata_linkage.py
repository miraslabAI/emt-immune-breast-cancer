"""Audit patient/sample linkage fields without exporting identifiers or outcomes."""
from __future__ import annotations

import os

import json
import re
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis/01_data_audit/singlecell_spatial/metadata_linkage_summary.json"
FILES = {
    "Integrated_BRCA_Atlas_2026": ROOT / "02_data/01_raw/single_cell/Integrated_BRCA_Atlas_2026/Integrated_BRCA_Global_Atlas_2026.h5ad",
    "GSE176078": ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad",
    "GSE225600_sc": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_sc_raw_counts.h5ad",
    "GSE225600_ST": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_ST_raw_counts.h5ad",
    "NeoTRIP_QUICHE": ROOT / "02_data/01_raw/spatial/NeoTRIP_QUICHE_14290163/nt_preprocessed.h5ad",
}
HTAN = ROOT / "02_data/01_raw/spatial/HTAN_Klughammer_2024_SlideSeq"
ID_RX = re.compile(r"(^|_)(patient|donor|subject|individual|case|sample|specimen|library)(_|$)", re.I)
BLOCK_RX = re.compile(r"outcome|response|survival|event|death|relapse|pcr|rcb|status|treatment", re.I)


def column_names(obs: h5py.Group) -> list[str]:
    return sorted(str(k) for k in obs.keys() if not str(k).startswith("_"))


def profile(node) -> dict:
    if isinstance(node, h5py.Group) and "codes" in node:
        codes = node["codes"][:]
        valid = codes[codes >= 0]
        return {"rows": int(len(codes)), "missing": int((codes < 0).sum()),
                "unique_nonmissing": int(np.unique(valid).size)}
    if isinstance(node, h5py.Dataset) and len(node.shape) == 1:
        values = node[:]
        if values.dtype.kind in "SUO":
            missing = np.fromiter((x in (b"", "", None) for x in values), dtype=bool, count=len(values))
        else:
            missing = np.isnan(values) if values.dtype.kind == "f" else np.zeros(len(values), dtype=bool)
        return {"rows": int(len(values)), "missing": int(missing.sum()),
                "unique_nonmissing": int(np.unique(values[~missing]).size)}
    return {"profiled": False, "reason": "unsupported_encoding"}


def inspect(name: str, path: Path, sealed: bool = False) -> dict:
    result = {"asset": name, "path": str(path), "sealed": sealed, "identifier_values_exported": False}
    with h5py.File(path, "r") as h:
        obs = h["obs"]
        cols = column_names(obs)
        candidates = [c for c in cols if ID_RX.search(c) and not BLOCK_RX.search(c)]
        result.update({"obs_column_count": len(cols), "obs_columns": cols, "linkage_candidates": candidates})
        if sealed:
            result["candidate_profiles"] = {}
            result["values_opened"] = False
        else:
            result["candidate_profiles"] = {c: profile(obs[c]) for c in candidates}
            result["values_opened"] = bool(candidates)
        result["patient_sample_linkage_possible"] = any("patient" in c.lower() or "donor" in c.lower() or
                                                         "subject" in c.lower() for c in candidates) and any(
                                                             "sample" in c.lower() or "specimen" in c.lower() for c in candidates)
    return result


def main() -> None:
    all_files = dict(FILES)
    all_files.update({f"HTAN_{p.stem}": p for p in sorted(HTAN.glob("*.h5ad"))})
    assets, failures = [], []
    for name, path in all_files.items():
        try:
            assets.append(inspect(name, path, sealed=(name == "NeoTRIP_QUICHE")))
        except Exception as exc:
            failures.append({"asset": name, "error": f"{type(exc).__name__}: {exc}"})
    summary = {
        "status": "PASS" if not failures else "FAIL", "assets": assets, "failures": failures,
        "identifiers_exported": False, "NeoTRIP_values_opened": False,
        "next_gate": "select_primary_single_cell_and_spatial_validation_roles",
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "assets": len(assets), "failures": failures,
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
