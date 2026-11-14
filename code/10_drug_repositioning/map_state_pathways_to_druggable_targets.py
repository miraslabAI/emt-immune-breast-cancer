#!/usr/bin/env python
"""Map state pathways to compartment-resolved ChEMBL targets and OmniPath context."""
from __future__ import annotations

import json
import os
import sqlite3
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
PATHWAYS = ROOT / "04_analysis/11_drug_repositioning/06_collapsed_state_pathway_map/state_pathway_representatives_annotated.tsv"
H5AD = ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad"
OMNIPATH = ROOT / "02_data/01_raw/reference/omnipath/omnipath_ligrecextra_human_2026-10-23.tsv"
CHEMBL = ROOT / "02_data/01_raw/reference/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/07_state_druggable_target_map"

CONFIDENCE_WEIGHT = {"very_high": 4, "high": 3, "moderate": 2, "exploratory": 1}
CHECKPOINT_GENES = {"PDCD1", "CD274", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "PDCD1LG2", "PVR", "NECTIN2"}
INHIBITORY_ACTIONS = ("INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE", "DEGRADER", "INACTIVATOR", "SUPPRESSOR")


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def categorical(obs: h5py.Group, column: str) -> np.ndarray:
    node = obs[column]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][...]
        categories = np.asarray(decode(node["categories"][...]), dtype=object)
        values = np.full(len(codes), "<MISSING>", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    if isinstance(node, h5py.Dataset):
        return np.asarray(decode(node[...]), dtype=str)
    raise ValueError(f"Unsupported obs encoding for {column}")


def read_h5ad_selected(path: Path, requested_genes: set[str]):
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        celltypes = categorical(obs, "celltype_major")
        var_key = handle["var"].attrs.get("_index", "_index")
        if isinstance(var_key, bytes):
            var_key = var_key.decode()
        genes = [value.upper() for value in decode(handle["var"][var_key][...])]
        lookup = {gene: index for index, gene in enumerate(genes)}
        selected_genes = sorted(requested_genes & set(lookup))
        selected_indices = [lookup[gene] for gene in selected_genes]
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            full = sparse.csr_matrix(node[...])
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            shape = tuple(int(value) for value in node.attrs["shape"])
            data, indices, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
            if encoding == "csr_matrix":
                full = sparse.csr_matrix((data, indices, indptr), shape=shape)
            elif encoding == "csc_matrix":
                full = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Unsupported X encoding: {encoding}")
        library = np.asarray(full.sum(axis=1)).ravel()
        selected = full[:, selected_indices].tocsr()
    return selected, library, celltypes, selected_genes


def compartment_group(celltype: str) -> str:
    text = celltype.lower()
    if "cancer epithelial" in text or "malignant" in text or "tumor" in text:
        return "malignant"
    if "normal epithelial" in text:
        return "normal_epithelial"
    if any(
        token in text
        for token in [
            "t-cell",
            "t cell",
            "b-cell",
            "b cell",
            "myeloid",
            "mast",
            "plasmablast",
            "plasma cell",
        ]
    ):
        return "immune"
    if any(token in text for token in ["caf", "fibro", "endothelial", "pvl", "perivascular", "stroma"]):
        return "stromal_vascular"
    return "other"


def compartment_expression(matrix, library, celltypes, genes):
    rows = []
    for celltype in sorted(np.unique(celltypes)):
        idx = np.flatnonzero(celltypes == celltype)
        gene_counts = np.asarray(matrix[idx].sum(axis=0)).ravel()
        total = float(library[idx].sum())
        cpm = gene_counts / total * 1e6 if total > 0 else np.zeros(len(genes))
        detection = np.asarray((matrix[idx] > 0).mean(axis=0)).ravel()
        for gene, value, fraction in zip(genes, cpm, detection):
            rows.append({
                "gene": gene, "celltype_major": celltype, "compartment": compartment_group(celltype),
                "cells": len(idx), "pseudobulk_cpm": float(value), "detection_fraction": float(fraction),
            })
    expression = pd.DataFrame(rows)
    grouped = expression.groupby(["gene", "compartment"], as_index=False).agg(
        pseudobulk_cpm=("pseudobulk_cpm", "sum"),
        maximum_detection_fraction=("detection_fraction", "max"),
    )
    top_rows = []
    for gene, group in grouped.groupby("gene"):
        group = group.sort_values("pseudobulk_cpm", ascending=False)
        top = group.iloc[0]
        second = float(group.iloc[1].pseudobulk_cpm) if len(group) > 1 else 0.0
        ratio = float((top.pseudobulk_cpm + 1e-6) / (second + 1e-6))
        assignment = str(top.compartment) if ratio >= 2.0 else "shared"
        top_rows.append({
            "gene": gene, "top_compartment": str(top.compartment),
            "assigned_compartment": assignment, "top_to_second_cpm_ratio": ratio,
            "top_compartment_cpm": float(top.pseudobulk_cpm),
        })
    return expression, pd.DataFrame(top_rows)


def chembl_columns(connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def query_chembl(genes: set[str]) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{CHEMBL.as_posix()}?mode=ro", uri=True)
    mechanism_columns = chembl_columns(connection, "drug_mechanism")
    max_phase_expr = "dm.max_phase" if "max_phase" in mechanism_columns else "md.max_phase"
    query = f"""
        SELECT DISTINCT
            cs.component_synonym AS gene,
            md.chembl_id AS molecule_chembl_id,
            md.pref_name AS molecule_name,
            md.max_phase AS molecule_max_phase,
            {max_phase_expr} AS mechanism_max_phase,
            dm.action_type,
            dm.mechanism_of_action,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.target_type
        FROM component_synonyms cs
        JOIN target_components tc ON tc.component_id = cs.component_id
        JOIN target_dictionary td ON td.tid = tc.tid
        JOIN drug_mechanism dm ON dm.tid = td.tid
        JOIN molecule_dictionary md ON md.molregno = dm.molregno
        WHERE cs.syn_type = 'GENE_SYMBOL' AND cs.component_synonym IN ({{placeholders}})
    """
    chunks = []
    genes = sorted(genes)
    for start in range(0, len(genes), 500):
        subset = genes[start:start + 500]
        sql = query.replace("{placeholders}", ",".join("?" for _ in subset))
        chunks.append(pd.read_sql_query(sql, connection, params=subset))
    connection.close()
    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True).drop_duplicates()
    result["gene"] = result.gene.astype(str).str.upper()
    return result


def strategy(roles: set[str]) -> str:
    if roles == {"immune_context_or_bottleneck"}:
        return "immune_direction_or_checkpoint_context_required"
    if "metabolic_dependency_or_adaptation" in roles or "stress_adaptation_candidate" in roles:
        return "dependency_validation_required_before_inhibition"
    if roles & {"signaling_driver_candidate", "barrier_plasticity_candidate",
                "microenvironmental_bottleneck_candidate", "vascular_stromal_context_candidate"}:
        return "inhibit_or_remodel_state_high_axis"
    return "mechanistic_direction_unresolved"


def main() -> None:
    for path in [PATHWAYS, H5AD, OMNIPATH, CHEMBL, LOCK]:
        if not path.exists():
            raise FileNotFoundError(path)
    pathways = pd.read_csv(PATHWAYS, sep="\t")
    pathways["genes"] = pathways.detected_gene_symbols.astype(str).str.split(";")
    all_genes = {gene for genes in pathways.genes for gene in genes if gene}
    chembl = query_chembl(all_genes)
    if chembl.empty:
        raise ValueError("No ChEMBL drug mechanisms mapped to pathway genes")
    target_genes = set(chembl.gene)

    matrix, library, celltypes, measured_targets = read_h5ad_selected(H5AD, target_genes)
    celltype_expression, compartments = compartment_expression(matrix, library, celltypes, measured_targets)

    exploded = pathways.explode("genes").rename(columns={"genes": "gene"})
    exploded = exploded.loc[exploded.gene.isin(target_genes)].copy()
    evidence_rows = []
    for (state, gene), group in exploded.groupby(["bulk_state_class", "gene"]):
        roles = set(group.provisional_intervention_role.astype(str))
        confidence = sorted(set(group.multiscale_confidence.astype(str)),
                            key=lambda value: -CONFIDENCE_WEIGHT.get(value, 0))
        evidence_rows.append({
            "state": state, "gene": gene, "representative_pathways": int(group.pathway.nunique()),
            "pathways": ";".join(sorted(group.pathway.unique())),
            "biological_themes": ";".join(sorted(set(group.biological_theme.astype(str)))),
            "provisional_roles": ";".join(sorted(roles)),
            "highest_multiscale_confidence": confidence[0],
            "pathway_evidence_weight": int(sum(CONFIDENCE_WEIGHT.get(value, 0) for value in group.multiscale_confidence)),
            "provisional_strategy": strategy(roles),
        })
    target_evidence = pd.DataFrame(evidence_rows).merge(compartments, on="gene", how="left")

    omni = pd.read_csv(OMNIPATH, sep="\t", dtype=str)
    omni.source_genesymbol = omni.source_genesymbol.astype(str).str.upper()
    omni.target_genesymbol = omni.target_genesymbol.astype(str).str.upper()
    omni_context = omni.loc[
        omni.source_genesymbol.isin(target_genes) | omni.target_genesymbol.isin(target_genes)
    ].copy()
    omni_context.to_csv(OUT / "omnipath_target_ligand_receptor_context.tsv", sep="\t", index=False) if OUT.exists() else None
    omni_rows = []
    for gene in target_genes:
        as_source = omni_context.loc[omni_context.source_genesymbol.eq(gene)]
        as_target = omni_context.loc[omni_context.target_genesymbol.eq(gene)]
        omni_rows.append({
            "gene": gene, "ligand_edges_as_source": len(as_source), "receptor_edges_as_target": len(as_target),
            "source_partners": ";".join(sorted(set(as_source.target_genesymbol))[:50]),
            "target_ligands": ";".join(sorted(set(as_target.source_genesymbol))[:50]),
        })
    target_evidence = target_evidence.merge(pd.DataFrame(omni_rows), on="gene", how="left")

    drugs = target_evidence.merge(chembl, on="gene", how="inner")
    drugs["inhibitory_mechanism"] = drugs.action_type.fillna("").astype(str).str.upper().map(
        lambda value: any(token in value for token in INHIBITORY_ACTIONS)
    )
    drugs["checkpoint_release_candidate"] = drugs.gene.isin(CHECKPOINT_GENES) & drugs.inhibitory_mechanism
    drugs["mechanism_direction_compatible"] = np.where(
        drugs.provisional_strategy.eq("inhibit_or_remodel_state_high_axis"),
        drugs.inhibitory_mechanism,
        np.where(drugs.checkpoint_release_candidate, True, False),
    )

    OUT.mkdir(parents=True, exist_ok=True)
    celltype_expression.to_csv(OUT / "target_celltype_expression.tsv.gz", sep="\t", index=False, compression="gzip")
    compartments.to_csv(OUT / "target_compartment_assignment.tsv", sep="\t", index=False)
    target_evidence.to_csv(OUT / "state_druggable_target_evidence.tsv", sep="\t", index=False)
    drugs.to_csv(OUT / "state_chembl_drug_mechanism_candidates.tsv.gz", sep="\t", index=False, compression="gzip")
    omni_context.to_csv(OUT / "omnipath_target_ligand_receptor_context.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "analysis_type": "state pathway to compartment-resolved ChEMBL target mapping",
        "representative_pathways": int(len(pathways)),
        "unique_pathway_genes": int(len(all_genes)),
        "genes_with_chembl_mechanisms": int(len(target_genes)),
        "targets_measured_in_gse176078": int(len(measured_targets)),
        "state_target_pairs": int(len(target_evidence)),
        "state_drug_mechanism_rows": int(len(drugs)),
        "unique_molecules": int(drugs.molecule_chembl_id.nunique()),
        "direction_compatible_rows": int(drugs.mechanism_direction_compatible.sum()),
        "compartment_assignments": {str(k): int(v) for k, v in compartments.assigned_compartment.value_counts().items()},
        "interpretation_constraint": "target membership and drug mechanism are nomination evidence, not proof of state dependency or efficacy",
        "next_gate": "state-specific LINCS reversal/preservation scoring plus PRISM/GDSC dependency validation",
    }
    (OUT / "state_druggable_target_map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
