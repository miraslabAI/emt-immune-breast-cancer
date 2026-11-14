"""Outcome-free structural/design audit for NeoTRIP spatial mechanism analysis."""

from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "02_data/01_raw/spatial/NeoTRIP_QUICHE_14290163/nt_preprocessed.h5ad"
OUT = ROOT / "04_analysis/01_data_audit/spatial_mechanism/neotrip_outcome_free"
ALLOWED = ["Patient_ID", "Tissue_ID", "fov", "BiopsyPhase", "cellAnnotation",
           "cell_cluster_broad", "isEpithelial", "centroid-0", "centroid-1"]
FORBIDDEN = ["pCR", "Arm"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in arr
        ], dtype=object)
    return arr


def read_column(obs, name):
    node = obs[name]
    if isinstance(node, h5py.Dataset):
        return decode(node[...])
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding == "categorical" or ("codes" in node and "categories" in node):
        codes = node["codes"][...].astype(int)
        categories = decode(node["categories"][...])
        values = np.full(len(codes), None, dtype=object)
        good = codes >= 0
        values[good] = categories[codes[good]]
        return values
    raise ValueError(f"Unsupported obs encoding for {name}: {encoding!r}")


def read_index(group):
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return decode(group[key][...])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, "r") as handle:
        obs = handle["obs"]
        missing = sorted(set(ALLOWED) - set(obs.keys()))
        if missing:
            raise ValueError(f"Missing allowed NeoTRIP metadata: {missing}")
        # Explicit allow-list: pCR and Arm are never accessed.
        frame = pd.DataFrame({name: read_column(obs, name) for name in ALLOWED})
        obs_names = read_index(obs)
        var_names = read_index(handle["var"])
        if "spatial" not in handle["obsm"]:
            raise ValueError("NeoTRIP obsm/spatial missing")
        spatial_node = handle["obsm"]["spatial"]
        if not isinstance(spatial_node, h5py.Dataset):
            raise ValueError("Unexpected NeoTRIP spatial encoding")
        spatial = spatial_node[...]

    if len(frame) != len(obs_names) or spatial.shape != (len(frame), 2):
        raise ValueError("NeoTRIP row/spatial dimension mismatch")
    if not np.isfinite(spatial).all():
        raise ValueError("Non-finite NeoTRIP spatial coordinates")
    if frame.Patient_ID.isna().any() or frame.fov.isna().any():
        raise ValueError("Missing patient or FOV identifier")

    phase_counts = frame.BiopsyPhase.value_counts(dropna=False).rename_axis("BiopsyPhase").reset_index(name="cells")
    annotation_counts = frame.cellAnnotation.value_counts(dropna=False).rename_axis("cellAnnotation").reset_index(name="cells")
    broad_counts = frame.cell_cluster_broad.value_counts(dropna=False).rename_axis("cell_cluster_broad").reset_index(name="cells")
    patient_phase = (frame.groupby(["BiopsyPhase", "Patient_ID"], dropna=False).size()
                     .rename("cells").reset_index())
    fov_summary = (frame.groupby(["Patient_ID", "Tissue_ID", "BiopsyPhase", "fov"], dropna=False)
                   .agg(cells=("fov", "size"), epithelial_cells=("isEpithelial", "sum"))
                   .reset_index())
    phase_counts.to_csv(OUT / "biopsy_phase_counts.tsv", sep="\t", index=False)
    annotation_counts.to_csv(OUT / "cell_annotation_counts.tsv", sep="\t", index=False)
    broad_counts.to_csv(OUT / "broad_cell_cluster_counts.tsv", sep="\t", index=False)
    patient_phase.to_csv(OUT / "patient_phase_cell_counts.tsv", sep="\t", index=False)
    fov_summary.to_csv(OUT / "fov_design_summary.tsv", sep="\t", index=False)
    pd.DataFrame({"marker": var_names}).to_csv(OUT / "available_markers.tsv", sep="\t", index=False)

    centroid = frame[["centroid-0", "centroid-1"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    agreement = None
    if np.isfinite(centroid).all():
        direct = np.max(np.abs(centroid - spatial))
        swapped = np.max(np.abs(centroid[:, ::-1] - spatial))
        agreement = {"direct_max_abs_delta": float(direct), "swapped_max_abs_delta": float(swapped)}
    result = {
        "status": "PASS", "audit": "NeoTRIP outcome-free spatial design",
        "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "rows": int(len(frame)), "markers": int(len(var_names)),
        "patients": int(frame.Patient_ID.nunique()), "tissues": int(frame.Tissue_ID.nunique()),
        "fovs": int(frame.fov.nunique()),
        "biopsy_phases": {str(k): int(v) for k, v in frame.BiopsyPhase.value_counts(dropna=False).items()},
        "epithelial_cells": int(pd.to_numeric(frame.isEpithelial, errors="coerce").fillna(0).sum()),
        "cell_annotations": {str(k): int(v) for k, v in frame.cellAnnotation.value_counts(dropna=False).items()},
        "broad_clusters": {str(k): int(v) for k, v in frame.cell_cluster_broad.value_counts(dropna=False).items()},
        "spatial_coordinate_shape": list(spatial.shape), "centroid_spatial_agreement": agreement,
        "allowed_columns_read": ALLOWED, "forbidden_columns_not_read": FORBIDDEN,
        "external_outcomes_used": False,
        "next_step": "lock baseline spatial neighborhoods and EMT-state immune proximity analysis",
    }
    (OUT / "neotrip_outcome_free_spatial_design_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
