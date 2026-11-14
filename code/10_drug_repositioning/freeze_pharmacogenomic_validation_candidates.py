#!/usr/bin/env python
"""Freeze LINCS candidates and breast-cell EMT states before viability testing.

No PRISM or GDSC response values are read in this stage.  Only compound and
cell-line metadata plus untreated CCLE expression are used.
"""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LINCS_RANK = ROOT / "04_analysis/11_drug_repositioning/09_state_specific_lincs_scoring/lincs_compound_state_rankings_all_times.tsv.gz"
LINCS_SUMMARY = ROOT / "04_analysis/11_drug_repositioning/09_state_specific_lincs_scoring/state_specific_lincs_scoring_summary.json"
CCLE_MODEL = ROOT / "02_data/01_raw/drug_sensitivity/depmap_ccle_24Q4/Model.csv"
CCLE_EXPR = ROOT / "02_data/01_raw/drug_sensitivity/depmap_ccle_24Q4/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
PRISM_PRIMARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-treatment-info.csv"
PRISM_SECONDARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/secondary_screen/prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv"
PRISM_PRIMARY_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/prism-repurposing-20q2-primary-screen-cell-line-info_breast_models.tsv"
PRISM_SECONDARY_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/prism-repurposing-20q2-secondary-screen-cell-line-info_breast_models.tsv"
GDSC_DRUGS = ROOT / "02_data/01_raw/drug_sensitivity/gdsc/annotation/screened_compounds_rel_8.5_inchi.csv"
GDSC_MODELS = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests/GDSC_breast_models.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/10_pharmacogenomic_validation_lock"

EPITHELIAL = ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "OCLN"]
MESENCHYMAL = ["VIM", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "FN1", "ITGA5"]
PRIMARY_TIME_H = 24.0


def normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    sd = frame.std(axis=0, ddof=0).replace(0, np.nan)
    return (frame - mean) / sd


def expression_column_map(path: Path, genes: list[str]) -> tuple[str, dict[str, str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    id_column = columns[0]
    mapping = {}
    for gene in genes:
        matches = [column for column in columns if column == gene or column.startswith(f"{gene} (")]
        if len(matches) != 1:
            raise ValueError(f"CCLE expression column mapping failed for {gene}: {matches}")
        mapping[gene] = matches[0]
    return id_column, mapping


def build_breast_states() -> pd.DataFrame:
    models = pd.read_csv(CCLE_MODEL, low_memory=False)
    breast = models[
        models["OncotreeLineage"].astype("string").str.casefold().eq("breast")
        & models["ModelID"].notna()
    ].copy()
    id_column, mapping = expression_column_map(CCLE_EXPR, EPITHELIAL + MESENCHYMAL)
    expression = pd.read_csv(CCLE_EXPR, usecols=[id_column] + list(mapping.values()), low_memory=False)
    expression = expression.rename(columns={value: key for key, value in mapping.items()})
    expression[id_column] = expression[id_column].astype(str)
    frame = breast.merge(expression, left_on="ModelID", right_on=id_column, how="inner", validate="one_to_one")
    genes = EPITHELIAL + MESENCHYMAL
    frame[genes] = frame[genes].apply(pd.to_numeric, errors="coerce")
    if frame[genes].isna().any().any():
        raise ValueError("Missing/non-numeric locked EMT genes in CCLE breast expression")
    gene_z = zscore(frame[genes])
    frame["epithelial_core"] = gene_z[EPITHELIAL].mean(axis=1)
    frame["mesenchymal_core"] = gene_z[MESENCHYMAL].mean(axis=1)
    frame["emt_position"] = frame["mesenchymal_core"] - frame["epithelial_core"]
    e_hi = frame["epithelial_core"].ge(frame["epithelial_core"].median())
    m_hi = frame["mesenchymal_core"].ge(frame["mesenchymal_core"].median())
    frame["emt_state"] = np.select(
        [e_hi & ~m_hi, e_hi & m_hi, ~e_hi & m_hi, ~e_hi & ~m_hi],
        ["epithelial-enriched", "hybrid-peak", "mesenchymal-enriched", "double-low"],
        default="ERROR",
    )
    keep = ["ModelID", "CCLEName", "StrippedCellLineName", "OncotreeSubtype",
            "ModelSubtypeFeatures", "epithelial_core", "mesenchymal_core",
            "emt_position", "emt_state"]
    return frame[keep].copy()


def freeze_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = pd.read_csv(LINCS_RANK, sep="\t", compression="gzip", low_memory=False)
    numeric = ["time_h", "n_cell_lines", "median_target_reversal", "median_composite_benefit",
               "cell_direction_consistency", "evidence_adjusted_score"]
    for column in numeric:
        ranks[column] = pd.to_numeric(ranks[column], errors="coerce")
    ranks["preservation_gate_pass"] = ranks["preservation_gate_pass"].astype(str).str.casefold().eq("true")
    eligible = ranks[
        np.isclose(ranks["time_h"], PRIMARY_TIME_H)
        & ranks["n_cell_lines"].ge(3)
        & ranks["median_target_reversal"].gt(0)
        & ranks["median_composite_benefit"].gt(0)
        & ranks["cell_direction_consistency"].ge(0.60)
        & ranks["preservation_gate_pass"]
    ].copy()
    eligible["normalized_name"] = eligible["pert_iname"].map(normal_name)
    # Consolidate duplicate Broad perturbation identifiers sharing a name.
    frozen = eligible.sort_values("evidence_adjusted_score", ascending=False).drop_duplicates(
        ["state", "normalized_name"], keep="first"
    )
    frozen["validation_role"] = np.where(
        frozen["state"].eq("epithelial-enriched"),
        "exploratory_dependency_test",
        "state_repair_viability_test",
    )
    frozen["candidate_frozen_before_response_access"] = True
    return frozen, ranks


def map_screen(name: str, info_path: Path, frozen: pd.DataFrame) -> pd.DataFrame:
    info = pd.read_csv(info_path, low_memory=False)
    info["normalized_name"] = info["name"].map(normal_name)
    info = info[info["normalized_name"].ne("") & info["normalized_name"].ne("na")]
    columns = [column for column in ["normalized_name", "name", "broad_id", "moa", "target", "smiles", "phase"] if column in info]
    info = info[columns].drop_duplicates()
    mapped = frozen.merge(info, on="normalized_name", how="left", suffixes=("_lincs", f"_{name}"))
    mapped["screen"] = name
    mapped["compound_mapped"] = mapped["broad_id"].notna()
    return mapped


def main() -> None:
    required = [LINCS_RANK, LINCS_SUMMARY, CCLE_MODEL, CCLE_EXPR, PRISM_PRIMARY_INFO,
                PRISM_SECONDARY_INFO, PRISM_PRIMARY_MODELS, PRISM_SECONDARY_MODELS,
                GDSC_DRUGS, GDSC_MODELS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    lincs_summary = json.loads(LINCS_SUMMARY.read_text(encoding="utf-8"))
    if lincs_summary.get("status") != "PASS":
        raise RuntimeError("LINCS scoring did not pass")

    states = build_breast_states()
    frozen, _ = freeze_candidates()
    if frozen.empty:
        raise RuntimeError("No LINCS candidates passed the frozen validation gate")

    primary_map = map_screen("PRISM_primary", PRISM_PRIMARY_INFO, frozen)
    secondary_map = map_screen("PRISM_secondary", PRISM_SECONDARY_INFO, frozen)

    gdsc = pd.read_csv(GDSC_DRUGS, low_memory=False)
    gdsc["normalized_name"] = gdsc["DRUG_NAME"].map(normal_name)
    gdsc_map = frozen.merge(
        gdsc[["normalized_name", "DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY", "CHEMBL_ID", "INCHI_KEY"]].drop_duplicates(),
        on="normalized_name", how="left",
    )
    gdsc_map["screen"] = "GDSC2"
    gdsc_map["compound_mapped"] = gdsc_map["DRUG_ID"].notna()

    prism_primary_models = pd.read_csv(PRISM_PRIMARY_MODELS, sep="\t", low_memory=False)
    prism_secondary_models = pd.read_csv(PRISM_SECONDARY_MODELS, sep="\t", low_memory=False)
    gdsc_models = pd.read_csv(GDSC_MODELS, sep="\t", low_memory=False)
    model_rows = []
    for screen, frame, model_column in [
        ("PRISM_primary", prism_primary_models, "depmap_id"),
        ("PRISM_secondary", prism_secondary_models, "depmap_id"),
        ("GDSC2", gdsc_models, "BROAD_ID"),
    ]:
        valid = frame[frame[model_column].notna()].copy()
        linked = valid.merge(states, left_on=model_column, right_on="ModelID", how="inner")
        for state, group in linked.groupby("emt_state"):
            model_rows.append({"screen": screen, "emt_state": state, "breast_models": int(group["ModelID"].nunique())})
    model_coverage = pd.DataFrame(model_rows)

    mappings = pd.concat([
        primary_map.assign(mapping_basis="normalized_name"),
        secondary_map.assign(mapping_basis="normalized_name"),
        gdsc_map.assign(mapping_basis="normalized_name"),
    ], ignore_index=True, sort=False)
    coverage_rows = []
    for (state, screen), group in mappings.groupby(["state", "screen"], sort=True):
        coverage_rows.append({
            "state": state,
            "screen": screen,
            "frozen_candidates": int(group["normalized_name"].nunique()),
            "mapped_candidates": int(group.loc[group["compound_mapped"], "normalized_name"].nunique()),
        })
    coverage = pd.DataFrame(coverage_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    states.to_csv(OUT / "frozen_ccle_breast_emt_states.tsv", sep="\t", index=False)
    frozen.to_csv(OUT / "frozen_lincs_candidates_for_viability.tsv", sep="\t", index=False)
    mappings.to_csv(OUT / "frozen_candidate_screen_mapping.tsv", sep="\t", index=False)
    coverage.to_csv(OUT / "frozen_candidate_screen_coverage.tsv", sep="\t", index=False)
    model_coverage.to_csv(OUT / "screen_breast_model_state_coverage.tsv", sep="\t", index=False)

    state_counts = states["emt_state"].value_counts().to_dict()
    candidate_counts = frozen.groupby("state")["normalized_name"].nunique().to_dict()
    mapped_counts = (
        mappings[mappings["compound_mapped"]]
        .groupby(["state", "screen"])["normalized_name"].nunique()
        .rename("count").reset_index()
    )
    failures = []
    for state in ["epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"]:
        if state_counts.get(state, 0) < 5:
            failures.append(f"insufficient_ccle_state_models:{state}:{state_counts.get(state, 0)}")

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "FAIL" if failures else "PASS",
        "analysis_type": "outcome-blind pharmacogenomic candidate and EMT-state freeze",
        "prism_response_values_loaded_or_used": False,
        "gdsc_response_values_loaded_or_used": False,
        "candidate_gate": "24h; >=3 LINCS breast lines; positive target reversal and composite; >=60% direction consistency; immune-preservation pass",
        "frozen_candidates_by_state": {str(k): int(v) for k, v in candidate_counts.items()},
        "ccle_breast_models": int(states["ModelID"].nunique()),
        "ccle_state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "mapped_candidate_counts": {
            f"{row.state}::{row.screen}": int(row.count) for row in mapped_counts.itertuples()
        },
        "state_definition": "median-quadrant coactivation using locked epithelial and mesenchymal core genes",
        "failures": failures,
        "next_gate": "test only frozen mapped candidates in PRISM primary/secondary and GDSC2; compare matched EMT state versus other breast states",
    }
    (OUT / "pharmacogenomic_validation_lock_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
