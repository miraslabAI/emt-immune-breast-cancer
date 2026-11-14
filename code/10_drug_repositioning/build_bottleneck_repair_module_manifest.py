#!/usr/bin/env python
"""Freeze compartment-aware bottleneck-repair modules before LINCS scoring."""
from __future__ import annotations

import os

import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HALLMARK = ROOT / "02_data/01_raw/reference/msigdb/h.all.v2026.1.Hs.symbols.gmt"
REACTOME = ROOT / "02_data/01_raw/reference/reactome/ReactomePathways.gmt.zip"
LINCS = ROOT / "02_data/01_raw/reference/lincs/GSE70138/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
MALIGNANT_H5AD = ROOT / "02_data/03_processed/single_cell/GSE176078/GSE176078_malignant_patient_metacells.h5ad"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/02_bottleneck_repair_modules"

CURATED = {
    "antigen_visibility": {
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "NLRC5",
        "PSMB8", "PSMB9", "PSMB10", "ERAP1", "ERAP2", "CALR", "CANX", "PDIA3",
    },
    "t_nk_recruitment": {
        "CXCL9", "CXCL10", "CXCL11", "CCL4", "CCL5", "XCL1", "XCL2", "FLT3LG", "IL15",
    },
    "checkpoint_ligand_pressure": {
        "CD274", "PDCD1LG2", "LGALS9", "PVR", "NECTIN2", "CD80", "CD86", "IDO1",
    },
    "immune_execution_partner": {
        "CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY",
        "IFNG", "TBX21", "EOMES", "NCR1", "KLRK1", "CTSW", "CCL5",
    },
}

HALLMARK_MAP = {
    "ifn_response": ["HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE"],
    "tgfb_ecm_barrier": ["HALLMARK_TGF_BETA_SIGNALING", "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
}

REACTOME_MAP = {
    "antigen_visibility": [
        "Antigen processing-Cross presentation",
        "Class I MHC mediated antigen processing & presentation",
    ],
    "ifn_response": ["Interferon gamma signaling", "Interferon Signaling"],
    "tgfb_ecm_barrier": [
        "Extracellular matrix organization",
        "Cell-extracellular matrix interactions",
        "Signaling by TGF-beta Receptor Complex",
        "TGF-beta receptor signaling activates SMADs",
        "TGF-beta receptor signaling in EMT (epithelial to mesenchymal transition)",
    ],
}

OBJECTIVES = [
    ("epithelial_cold", "antigen_visibility", "induce", "malignant", "primary"),
    ("epithelial_cold", "ifn_response", "induce", "malignant", "primary"),
    ("epithelial_cold", "t_nk_recruitment", "induce", "malignant", "primary"),
    ("epithelial_cold", "tgfb_ecm_barrier", "avoid_induction", "malignant_stromal", "constraint"),
    ("hybrid_visible_barrier", "tgfb_ecm_barrier", "suppress", "malignant_stromal", "primary"),
    ("hybrid_visible_barrier", "antigen_visibility", "preserve_or_induce", "malignant", "constraint"),
    ("hybrid_visible_barrier", "ifn_response", "preserve_or_induce", "malignant", "constraint"),
    ("hybrid_visible_barrier", "t_nk_recruitment", "preserve_or_induce", "malignant", "constraint"),
    ("mesenchymal_engaged_incomplete", "tgfb_ecm_barrier", "suppress", "malignant_stromal", "primary"),
    ("mesenchymal_engaged_incomplete", "checkpoint_ligand_pressure", "suppress", "malignant", "primary"),
    ("mesenchymal_engaged_incomplete", "antigen_visibility", "preserve_or_induce", "malignant", "constraint"),
    ("mesenchymal_engaged_incomplete", "immune_execution_partner", "induce_or_release_inhibition", "immune", "partner"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gmt_lines(lines) -> dict[str, set[str]]:
    result = {}
    for raw in lines:
        parts = raw.rstrip("\n\r").split("\t")
        if len(parts) >= 3:
            result[parts[0]] = {gene.upper() for gene in parts[2:] if gene}
    return result


def read_reactome(path: Path) -> dict[str, set[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one Reactome GMT member, observed {names}")
        with archive.open(names[0]) as handle:
            lines = (line.decode("utf-8") for line in handle)
            return read_gmt_lines(lines)


def h5ad_genes(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        key = handle["var"].attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        values = handle["var"][key][...]
    return {
        (value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)).upper()
        for value in values
    }


def main() -> None:
    for path in [HALLMARK, REACTOME, LINCS, MALIGNANT_H5AD, LOCK]:
        if not path.exists():
            raise FileNotFoundError(path)
    hallmark = read_gmt_lines(HALLMARK.read_text(encoding="utf-8").splitlines())
    reactome = read_reactome(REACTOME)
    malignant_genes = h5ad_genes(MALIGNANT_H5AD)
    lincs = pd.read_csv(LINCS, sep="\t", compression="gzip", dtype=str)
    lincs["gene"] = lincs.pr_gene_symbol.astype(str).str.upper()
    bing = set(lincs.loc[pd.to_numeric(lincs.pr_is_bing, errors="coerce").eq(1), "gene"])
    landmark = set(lincs.loc[pd.to_numeric(lincs.pr_is_lm, errors="coerce").eq(1), "gene"])

    modules = {name: set(genes) for name, genes in CURATED.items()}
    sources: dict[str, dict[str, set[str]]] = {
        name: {"curated_core": set(genes)} for name, genes in CURATED.items()
    }
    for module, names in HALLMARK_MAP.items():
        modules.setdefault(module, set())
        sources.setdefault(module, {})
        for name in names:
            if name not in hallmark:
                raise KeyError(f"Missing Hallmark set: {name}")
            modules[module].update(hallmark[name])
            sources[module][f"MSigDB::{name}"] = hallmark[name]
    for module, names in REACTOME_MAP.items():
        modules.setdefault(module, set())
        sources.setdefault(module, {})
        for name in names:
            if name not in reactome:
                raise KeyError(f"Missing Reactome set: {name}")
            modules[module].update(reactome[name])
            sources[module][f"Reactome::{name}"] = reactome[name]

    rows = []
    for module, genes in sorted(modules.items()):
        for gene in sorted(genes):
            gene_sources = sorted(source for source, members in sources[module].items() if gene in members)
            rows.append({
                "module": module,
                "gene": gene,
                "sources": ";".join(gene_sources),
                "malignant_detected": gene in malignant_genes,
                "lincs_bing": gene in bing,
                "lincs_landmark": gene in landmark,
                "malignant_lincs_query_eligible": gene in malignant_genes and gene in bing,
            })
    manifest = pd.DataFrame(rows)
    summary_rows = []
    failures = []
    for module, group in manifest.groupby("module"):
        eligible = int(group.malignant_lincs_query_eligible.sum())
        summary_rows.append({
            "module": module,
            "genes": int(len(group)),
            "malignant_detected": int(group.malignant_detected.sum()),
            "lincs_bing": int(group.lincs_bing.sum()),
            "malignant_lincs_query_eligible": eligible,
        })
        minimum = 5 if module in {"t_nk_recruitment", "checkpoint_ligand_pressure"} else 10
        if module != "immune_execution_partner" and eligible < minimum:
            failures.append(f"insufficient_malignant_lincs_coverage:{module}:{eligible}<{minimum}")

    objectives = pd.DataFrame(OBJECTIVES, columns=["emt_immune_state", "module", "desired_action", "compartment", "role"])
    if not set(objectives.module).issubset(modules):
        failures.append("objective_references_unknown_module")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "bottleneck_repair_gene_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "bottleneck_repair_module_coverage.tsv", sep="\t", index=False)
    objectives.to_csv(OUT / "state_bottleneck_repair_objectives.tsv", sep="\t", index=False)
    summary = {
        "status": "FAIL" if failures else "PASS",
        "analysis_type": "outcome-free compartment-aware bottleneck-repair module freeze",
        "whole_state_reversal_primary": False,
        "treatment_outcomes_used": False,
        "modules": {row["module"]: row["genes"] for row in summary_rows},
        "malignant_lincs_query_eligible": {
            row["module"]: row["malignant_lincs_query_eligible"] for row in summary_rows
        },
        "repair_objectives": int(len(objectives)),
        "failures": failures,
        "input_sha256": {str(path): sha256(path) for path in [HALLMARK, REACTOME, LINCS, MALIGNANT_H5AD, LOCK]},
        "next_gate": "validate module directions across malignant single-cell, bulk cohorts and spatial evidence before LINCS compound scoring",
    }
    (OUT / "bottleneck_repair_module_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
