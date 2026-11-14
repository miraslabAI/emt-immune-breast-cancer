#!/usr/bin/env python
"""Audit cellular origin of compact therapy-panel genes in GSE176078 scRNA-seq."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
H5AD = ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad"
PANEL = ROOT / "04_analysis/11_drug_repositioning/22_compact_therapy_biomarker_panels/locked_compact_panel_genes.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/24_compact_panel_cellular_origin"


def decode(values):
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]


def categorical(obs: h5py.Group, column: str) -> np.ndarray:
    node = obs[column]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][...]
        categories = np.asarray(decode(node["categories"][...]), dtype=object)
        values = np.full(len(codes), "<MISSING>", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return np.asarray(decode(node[...]), dtype=str)


def compartment(celltype: str) -> str:
    text = celltype.lower()
    if "cancer epithelial" in text or "malignant" in text or "tumor" in text:
        return "malignant"
    if "normal epithelial" in text:
        return "normal_epithelial"
    if any(x in text for x in ["t-cell", "t cell", "b-cell", "b cell", "myeloid", "mast", "plasma", "plasmablast"]):
        return "immune"
    if any(x in text for x in ["caf", "fibro", "endothelial", "pvl", "perivascular", "stroma"]):
        return "stromal_vascular"
    return "other"


def load_selected(requested: set[str]):
    with h5py.File(H5AD, "r") as handle:
        celltypes = categorical(handle["obs"], "celltype_major")
        var_key = handle["var"].attrs.get("_index", "_index")
        if isinstance(var_key, bytes):
            var_key = var_key.decode()
        genes = [x.upper() for x in decode(handle["var"][var_key][...])]
        lookup = {g: i for i, g in enumerate(genes)}
        measured = sorted(requested & set(lookup))
        indices = [lookup[g] for g in measured]
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            full = sparse.csr_matrix(node[...])
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            shape = tuple(int(x) for x in node.attrs["shape"])
            data, ix, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
            if encoding == "csr_matrix":
                full = sparse.csr_matrix((data, ix, indptr), shape=shape)
            elif encoding == "csc_matrix":
                full = sparse.csc_matrix((data, ix, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Unsupported X encoding: {encoding}")
        library = np.asarray(full.sum(axis=1)).ravel()
        selected = full[:, indices].tocsr()
    return selected, library, celltypes, measured


def aggregate(matrix, library, labels, genes, level):
    rows = []
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        counts = np.asarray(matrix[idx].sum(axis=0)).ravel()
        total = float(library[idx].sum())
        cpm = counts / total * 1e6 if total else np.zeros(len(genes))
        detected = np.asarray((matrix[idx] > 0).mean(axis=0)).ravel()
        for gene, value, fraction in zip(genes, cpm, detected):
            rows.append({"level": level, "group": label, "gene": gene, "cells": len(idx),
                         "pseudobulk_cpm": float(value), "detection_fraction": float(fraction)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL, sep="\t")
    panel["gene"] = panel["gene"].str.upper()
    requested = set(panel["gene"])
    matrix, library, celltypes, measured = load_selected(requested)
    missing = sorted(requested - set(measured))
    if len(measured) / len(requested) < 0.90:
        raise ValueError(f"Panel coverage below 90%: {len(measured)}/{len(requested)}; missing={missing}")
    compartments = np.asarray([compartment(x) for x in celltypes])
    by_celltype = aggregate(matrix, library, celltypes, measured, "celltype_major")
    by_compartment = aggregate(matrix, library, compartments, measured, "compartment")
    by_celltype.to_csv(OUT / "panel_gene_expression_by_celltype.tsv.gz", sep="\t", index=False, compression="gzip")
    by_compartment.to_csv(OUT / "panel_gene_expression_by_compartment.tsv", sep="\t", index=False)

    origin_rows = []
    for gene, group in by_compartment.groupby("gene"):
        group = group.sort_values("pseudobulk_cpm", ascending=False)
        top = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        ratio = float((top.pseudobulk_cpm + 1e-6) / ((second.pseudobulk_cpm if second is not None else 0) + 1e-6))
        origin_rows.append({
            "gene": gene, "top_compartment": top["group"],
            "top_to_second_cpm_ratio": ratio,
            "origin_call": top["group"] if ratio >= 2 else "shared",
            "top_cpm": float(top.pseudobulk_cpm),
            "top_detection_fraction": float(top.detection_fraction),
        })
    origins = pd.DataFrame(origin_rows)
    origins.to_csv(OUT / "panel_gene_cellular_origin.tsv", sep="\t", index=False)

    merged = panel.merge(origins, on="gene", how="left", validate="many_to_one")
    domain_rows = []
    for domain, group in merged.groupby("domain"):
        counts = group["origin_call"].value_counts()
        for origin in ["malignant", "stromal_vascular", "immune", "normal_epithelial", "other", "shared"]:
            domain_rows.append({
                "domain": domain, "origin": origin, "genes": len(group),
                "n": int(counts.get(origin, 0)), "fraction": float(counts.get(origin, 0) / len(group)),
            })
    domain_origin = pd.DataFrame(domain_rows)
    domain_origin.to_csv(OUT / "domain_cellular_origin_composition.tsv", sep="\t", index=False)

    # Ecosystem interpretation is prespecified; a stromal/immune origin is expected
    # for barrier/visibility domains, but not sufficient to call malignant EMT.
    interpretation = {
        "hybrid_state": "must retain malignant contribution; same-cell confirmation still required",
        "erbb_dependency": "prefer malignant origin",
        "akt_pathway": "RNA is not an activation assay; phosphoprotein required",
        "mtor_pathway": "RNA is not an activation assay; phosphoprotein required",
        "mesenchymal_state": "malignant contribution required to call tumor EMT; stromal dominance indicates CAF admixture",
        "vascular_stromal_barrier": "stromal/vascular origin is mechanistically expected",
        "immune_visibility": "immune and/or malignant antigen-presentation origin is expected",
        "incomplete_completion": "immune/myeloid/stromal origin is mechanistically expected",
    }
    interp_rows = []
    for domain, text in interpretation.items():
        subset = domain_origin[domain_origin["domain"] == domain]
        dominant = subset.sort_values(["n", "origin"], ascending=[False, True]).iloc[0]
        interp_rows.append({"domain": domain, "dominant_origin_call": dominant["origin"],
                            "dominant_fraction": dominant["fraction"], "interpretation_rule": text})
    pd.DataFrame(interp_rows).to_csv(OUT / "domain_origin_interpretation.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind single-cell cellular-origin audit of compact therapy panel",
        "cells": int(matrix.shape[0]),
        "celltypes": int(len(np.unique(celltypes))),
        "requested_unique_genes": len(requested),
        "measured_unique_genes": len(measured),
        "missing_genes": missing,
        "important_guardrail": "bulk mesenchymal score is not called tumor-intrinsic EMT when its compact genes are stromal-dominant",
        "outcomes_used": False,
        "next_gate": "separate malignant EMT qualification from ecosystem barrier/immune qualification and revise the therapy selector if required",
    }
    (OUT / "compact_panel_cellular_origin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
