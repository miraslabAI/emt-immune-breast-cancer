"""Low-memory structural audit of single-cell and spatial H5AD assets.

Only HDF5 structure, shapes, and key names are read. No obs values or outcomes
are loaded, which preserves the outcome firewall for treatment-linked cohorts.
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import h5py


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis" / "01_data_audit" / "singlecell_spatial"
FILES = {
    "Integrated_BRCA_Atlas_2026": ROOT / "02_data/01_raw/single_cell/Integrated_BRCA_Atlas_2026/Integrated_BRCA_Global_Atlas_2026.h5ad",
    "GSE176078_raw": ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad",
    "GSE225600_sc": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_sc_raw_counts.h5ad",
    "GSE225600_ST": ROOT / "02_data/02_interim/single_cell_spatial/GSE225600/GSE225600_ST_raw_counts.h5ad",
    "NeoTRIP_QUICHE": ROOT / "02_data/01_raw/spatial/NeoTRIP_QUICHE_14290163/nt_preprocessed.h5ad",
}
HTAN = ROOT / "02_data/01_raw/spatial/HTAN_Klughammer_2024_SlideSeq"


def group_keys(h: h5py.File, key: str) -> list[str]:
    return sorted(map(str, h[key].keys())) if key in h and isinstance(h[key], h5py.Group) else []


def frame_length(h: h5py.File, key: str) -> int | None:
    if key not in h or not isinstance(h[key], h5py.Group):
        return None
    g = h[key]
    idx = g.attrs.get("_index", "_index")
    if isinstance(idx, bytes):
        idx = idx.decode()
    return int(g[idx].shape[0]) if idx in g and hasattr(g[idx], "shape") else None


def inspect(name: str, path: Path, firewall: bool = False) -> dict:
    result = {"asset": name, "path": str(path), "exists": path.is_file(),
              "bytes": path.stat().st_size if path.is_file() else 0,
              "obs_values_opened": False, "outcome_values_opened": False,
              "firewall_protected": firewall}
    if not path.is_file():
        return result
    try:
        with h5py.File(path, "r") as h:
            result.update({
                "n_obs": frame_length(h, "obs"), "n_vars": frame_length(h, "var"),
                "top_level_keys": sorted(map(str, h.keys())),
                "layer_keys": group_keys(h, "layers"),
                "obsm_keys": group_keys(h, "obsm"),
                "obsp_keys": group_keys(h, "obsp"),
                "uns_key_count": len(group_keys(h, "uns")),
                "structurally_readable": True,
            })
    except Exception as exc:
        result.update({"structurally_readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = [inspect(n, p, firewall=(n == "NeoTRIP_QUICHE")) for n, p in FILES.items()]
    htan_files = sorted(HTAN.glob("*.h5ad")) if HTAN.exists() else []
    assets.extend(inspect(f"HTAN_{p.stem}", p) for p in htan_files)
    failures = [x["asset"] for x in assets if not x["exists"] or not x.get("structurally_readable", False)]
    summary = {
        "audit": "singlecell_spatial_structural",
        "method": "h5py_structure_only_no_obs_values",
        "asset_count": len(assets), "assets": assets, "failures": failures,
        "outcome_firewall_preserved": all(not x["outcome_values_opened"] for x in assets),
        "status": "PASS" if not failures else "FAIL",
        "next_gate": "metadata_dictionary_and_patient_sample_linkage_audit",
    }
    (OUT / "singlecell_spatial_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "assets": len(assets),
                      "failures": failures, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
