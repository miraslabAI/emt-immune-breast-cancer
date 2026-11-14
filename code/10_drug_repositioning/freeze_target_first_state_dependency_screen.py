#!/usr/bin/env python
"""Freeze a target-first state-dependency screen before broad response mining.

The candidate universe is derived only from frozen state pathways,
compartment-resolved targets, ChEMBL mechanisms, and screen metadata.  No
PRISM/GDSC response values are opened here.
"""
from __future__ import annotations

import os

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
TARGET_EVIDENCE = ROOT / "04_analysis/11_drug_repositioning/07_state_druggable_target_map/state_druggable_target_evidence.tsv"
CHEMBL_CANDIDATES = ROOT / "04_analysis/11_drug_repositioning/07_state_druggable_target_map/state_chembl_drug_mechanism_candidates.tsv.gz"
CHEMBL_DB = ROOT / "02_data/01_raw/reference/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
PRISM_PRIMARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/primary_screen/prism-repurposing-20q2-primary-screen-replicate-collapsed-treatment-info.csv"
PRISM_SECONDARY_INFO = ROOT / "02_data/01_raw/drug_sensitivity/depmap_prism/secondary_screen/prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv"
GDSC_DRUGS = ROOT / "02_data/01_raw/drug_sensitivity/gdsc/annotation/screened_compounds_rel_8.5_inchi.csv"
PRIOR_VALIDATION = ROOT / "04_analysis/11_drug_repositioning/11_pharmacogenomic_validation/state_specific_pharmacogenomic_synthesis.tsv"
OUT = ROOT / "04_analysis/11_drug_repositioning/12_target_first_dependency_lock"

ALLOWED_COMPARTMENTS = {"malignant", "shared"}
MIN_PHASE = 1


def normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().eq("true")


def chunked(values: list[str], size: int = 700):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def chembl_aliases(chembl_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure_rows = []
    synonym_rows = []
    connection = sqlite3.connect(f"file:{CHEMBL_DB.as_posix()}?mode=ro", uri=True)
    try:
        for block in chunked(chembl_ids):
            marks = ",".join("?" for _ in block)
            structure_rows.extend(connection.execute(
                f"""
                SELECT md.chembl_id, md.pref_name, cs.standard_inchi_key
                FROM molecule_dictionary md
                LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
                WHERE md.chembl_id IN ({marks})
                """, block
            ).fetchall())
            synonym_rows.extend(connection.execute(
                f"""
                SELECT md.chembl_id, ms.synonyms
                FROM molecule_dictionary md
                JOIN molecule_synonyms ms ON ms.molregno = md.molregno
                WHERE md.chembl_id IN ({marks})
                """, block
            ).fetchall())
    finally:
        connection.close()
    structure = pd.DataFrame(structure_rows, columns=["molecule_chembl_id", "chembl_pref_name", "standard_inchi_key"])
    synonyms = pd.DataFrame(synonym_rows, columns=["molecule_chembl_id", "alias"])
    return structure, synonyms


def select_mechanisms() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_EVIDENCE, sep="\t", low_memory=False)
    targets = targets[targets["assigned_compartment"].isin(ALLOWED_COMPARTMENTS)].copy()
    eligible_target = (
        targets["state"].eq("epithelial-enriched")
        & targets["provisional_strategy"].isin([
            "dependency_validation_required_before_inhibition",
            "inhibit_or_remodel_state_high_axis",
        ])
    ) | (
        targets["state"].isin(["hybrid-peak", "mesenchymal-enriched"])
        & targets["provisional_strategy"].eq("inhibit_or_remodel_state_high_axis")
    )
    targets = targets[eligible_target].copy()

    drugs = pd.read_csv(CHEMBL_CANDIDATES, sep="\t", compression="gzip", low_memory=False)
    drugs["molecule_max_phase"] = pd.to_numeric(drugs["molecule_max_phase"], errors="coerce")
    drugs = drugs[
        drugs["molecule_max_phase"].ge(MIN_PHASE)
        & as_bool(drugs["inhibitory_mechanism"])
    ].copy()
    # Epithelial metabolic/growth programs are explicitly dependency tests;
    # direction compatibility cannot be assumed before the viability screen.
    direction_ok = drugs["state"].eq("epithelial-enriched") | as_bool(drugs["mechanism_direction_compatible"])
    drugs = drugs[direction_ok]
    keys = targets[["state", "gene"]].drop_duplicates()
    drugs = drugs.merge(keys, on=["state", "gene"], how="inner", validate="many_to_one")
    return drugs


def alias_table(mechanisms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chembl_ids = sorted(mechanisms["molecule_chembl_id"].dropna().astype(str).unique())
    structure, synonyms = chembl_aliases(chembl_ids)
    preferred = mechanisms[["molecule_chembl_id", "molecule_name"]].drop_duplicates().rename(columns={"molecule_name": "alias"})
    chembl_preferred = structure[["molecule_chembl_id", "chembl_pref_name"]].rename(columns={"chembl_pref_name": "alias"})
    aliases = pd.concat([preferred, chembl_preferred, synonyms], ignore_index=True)
    aliases = aliases.dropna(subset=["alias"])
    aliases["normalized_alias"] = aliases["alias"].map(normal_name)
    aliases = aliases[aliases["normalized_alias"].ne("")].drop_duplicates()
    return aliases, structure


def map_prism(path: Path, screen: str, mechanisms: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    info = pd.read_csv(path, low_memory=False)
    info["normalized_alias"] = info["name"].map(normal_name)
    mapped = info.merge(aliases, on="normalized_alias", how="inner")
    evidence = mechanisms.groupby(["state", "molecule_chembl_id"], as_index=False).agg(
        state_target_genes=("gene", lambda x: ";".join(sorted(set(map(str, x))))),
        pathway_themes=("biological_themes", lambda x: ";".join(sorted(set(map(str, x))))),
        provisional_strategies=("provisional_strategy", lambda x: ";".join(sorted(set(map(str, x))))),
        chembl_actions=("action_type", lambda x: ";".join(sorted(set(map(str, x))))),
        maximum_clinical_phase=("molecule_max_phase", "max"),
    )
    mapped = mapped.merge(evidence, on="molecule_chembl_id", how="inner")
    mapped["screen"] = screen
    mapped["screen_drug_id"] = mapped["broad_id"].astype(str)
    mapped["screen_drug_name"] = mapped["name"].astype(str)
    mapped["mapping_basis"] = "ChEMBL preferred name or synonym"
    keep = ["state", "screen", "screen_drug_id", "screen_drug_name", "molecule_chembl_id",
            "state_target_genes", "pathway_themes", "provisional_strategies", "chembl_actions",
            "maximum_clinical_phase", "mapping_basis"]
    return mapped[keep].drop_duplicates()


def map_gdsc(mechanisms: pd.DataFrame, aliases: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    gdsc = pd.read_csv(GDSC_DRUGS, low_memory=False)
    gdsc["normalized_alias"] = gdsc["DRUG_NAME"].map(normal_name)
    direct = gdsc.merge(
        mechanisms[["molecule_chembl_id"]].drop_duplicates(),
        left_on="CHEMBL_ID", right_on="molecule_chembl_id", how="inner",
    )
    direct["mapping_basis"] = "CHEMBL_ID"
    inchi = gdsc.merge(
        structure.dropna(subset=["standard_inchi_key"]),
        left_on="INCHI_KEY", right_on="standard_inchi_key", how="inner",
    )
    inchi["mapping_basis"] = "standard InChIKey"
    named = gdsc.merge(aliases, on="normalized_alias", how="inner")
    named["mapping_basis"] = "ChEMBL preferred name or synonym"
    mapped = pd.concat([direct, inchi, named], ignore_index=True, sort=False)
    evidence = mechanisms.groupby(["state", "molecule_chembl_id"], as_index=False).agg(
        state_target_genes=("gene", lambda x: ";".join(sorted(set(map(str, x))))),
        pathway_themes=("biological_themes", lambda x: ";".join(sorted(set(map(str, x))))),
        provisional_strategies=("provisional_strategy", lambda x: ";".join(sorted(set(map(str, x))))),
        chembl_actions=("action_type", lambda x: ";".join(sorted(set(map(str, x))))),
        maximum_clinical_phase=("molecule_max_phase", "max"),
    )
    mapped = mapped.merge(evidence, on="molecule_chembl_id", how="inner")
    mapped["screen"] = "GDSC2"
    mapped["screen_drug_id"] = mapped["DRUG_ID"].astype(str)
    mapped["screen_drug_name"] = mapped["DRUG_NAME"].astype(str)
    keep = ["state", "screen", "screen_drug_id", "screen_drug_name", "molecule_chembl_id",
            "state_target_genes", "pathway_themes", "provisional_strategies", "chembl_actions",
            "maximum_clinical_phase", "mapping_basis"]
    return mapped[keep].drop_duplicates()


def collapse_universe(frame: pd.DataFrame) -> pd.DataFrame:
    frame["normalized_screen_name"] = frame["screen_drug_name"].map(normal_name)
    return frame.groupby(
        ["state", "screen", "screen_drug_id", "screen_drug_name", "normalized_screen_name"],
        as_index=False,
    ).agg(
        molecule_chembl_ids=("molecule_chembl_id", lambda x: ";".join(sorted(set(map(str, x))))),
        state_target_genes=("state_target_genes", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        pathway_themes=("pathway_themes", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        provisional_strategies=("provisional_strategies", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        chembl_actions=("chembl_actions", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        maximum_clinical_phase=("maximum_clinical_phase", "max"),
        mapping_bases=("mapping_basis", lambda x: ";".join(sorted(set(map(str, x))))),
    )


def main() -> None:
    required = [TARGET_EVIDENCE, CHEMBL_CANDIDATES, CHEMBL_DB, PRISM_PRIMARY_INFO,
                PRISM_SECONDARY_INFO, GDSC_DRUGS, PRIOR_VALIDATION]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    mechanisms = select_mechanisms()
    if mechanisms.empty:
        raise RuntimeError("No target-first mechanisms survived the frozen rules")
    aliases, structure = alias_table(mechanisms)
    mappings = pd.concat([
        map_prism(PRISM_PRIMARY_INFO, "PRISM_primary", mechanisms, aliases),
        map_prism(PRISM_SECONDARY_INFO, "PRISM_secondary", mechanisms, aliases),
        map_gdsc(mechanisms, aliases, structure),
    ], ignore_index=True, sort=False)
    universe = collapse_universe(mappings)

    prior = pd.read_csv(PRIOR_VALIDATION, sep="\t", low_memory=False)
    prior["normalized_screen_name"] = prior["candidate"].map(normal_name)
    prior_keys = set(zip(prior["state"], prior["normalized_screen_name"]))
    universe["previous_lincs_shortlist_candidate"] = [
        (state, name) in prior_keys
        for state, name in zip(universe["state"], universe["normalized_screen_name"])
    ]
    universe["response_values_seen_for_this_broad_universe"] = False

    OUT.mkdir(parents=True, exist_ok=True)
    mechanisms.to_csv(OUT / "target_first_eligible_chembl_mechanisms.tsv.gz", sep="\t", index=False, compression="gzip")
    universe.to_csv(OUT / "frozen_target_first_screen_universe.tsv", sep="\t", index=False)
    counts = universe.groupby(["state", "screen"], as_index=False).agg(
        screen_drugs=("screen_drug_id", "nunique"),
        target_genes=("state_target_genes", lambda x: len(set(";".join(map(str, x)).split(";")))),
        clinical_phase_3_or_4=("maximum_clinical_phase", lambda x: int((pd.to_numeric(x, errors="coerce") >= 3).sum())),
    )
    counts.to_csv(OUT / "target_first_screen_universe_counts.tsv", sep="\t", index=False)

    failures = []
    for state in ["epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"]:
        if state not in set(universe["state"]):
            failures.append(f"no_mapped_target_first_drugs:{state}")
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "FAIL" if failures else "PASS",
        "analysis_type": "response-blind target-first state-dependency universe freeze",
        "prism_response_values_loaded_or_used": False,
        "gdsc_response_values_loaded_or_used": False,
        "eligible_state_target_pairs": int(mechanisms[["state", "gene"]].drop_duplicates().shape[0]),
        "eligible_chembl_molecules": int(mechanisms["molecule_chembl_id"].nunique()),
        "frozen_state_screen_drug_rows": int(len(universe)),
        "screen_drugs_by_state_and_screen": {
            f"{row.state}::{row.screen}": int(row.screen_drugs) for row in counts.itertuples()
        },
        "epithelial_rule": "target-first dependency discovery; inhibition direction not assumed from expression alone",
        "hybrid_mesenchymal_rule": "direction-compatible inhibitory mechanisms against malignant/shared barrier or driver targets",
        "multiple_testing_rule": "BH correction across every frozen drug-state test, with screen-specific replication reported separately",
        "failures": failures,
        "next_gate": "open response values only for this frozen universe and identify replicated target-linked state dependencies",
    }
    (OUT / "target_first_dependency_lock_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
