"""Outcome-blind cell-line and compound identifier overlap across four screens."""
from __future__ import annotations

import os

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "02_data/01_raw/drug_sensitivity"
MAN = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests"
OUT = ROOT / "04_analysis/01_data_audit/pharmacogenomics/identifier_harmonization_summary.json"


def norm(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower()) if pd.notna(x) else ""


def values(df: pd.DataFrame, col: str) -> set[str]:
    return {norm(x) for x in df[col] if norm(x)} if col in df else set()


def pairwise(sets: dict[str, set[str]]) -> list[dict]:
    names = list(sets)
    return [{"a": names[i], "b": names[j], "overlap": len(sets[names[i]] & sets[names[j]]),
             "a_total": len(sets[names[i]]), "b_total": len(sets[names[j]])}
            for i in range(len(names)) for j in range(i + 1, len(names))]


def main() -> None:
    ccle = pd.read_csv(MAN / "DepMap_CCLE_breast_models.tsv", low_memory=False, sep="\t")
    gdsc = pd.read_csv(MAN / "GDSC_breast_models.tsv", low_memory=False, sep="\t")
    prism_files = sorted(MAN.glob("prism-*_breast_models.tsv"))
    prism = pd.concat([pd.read_csv(p, sep="\t", low_memory=False) for p in prism_files], ignore_index=True).drop_duplicates()
    ctrp = pd.read_csv(MAN / "CTRPv2_2_breast_cell_lines.tsv", sep="\t", low_memory=False)
    line_sets = {
        "CCLE": values(ccle, "StrippedCellLineName") | values(ccle, "CCLEName"),
        "GDSC": values(gdsc, "model_name") | values(gdsc, "CCLE_ID"),
        "PRISM": values(prism, "ccle_name"),
        "CTRP": values(ctrp, "ccl_name"),
    }
    gdrug = pd.read_csv(RAW / "gdsc/annotation/screened_compounds_rel_8.5_inchi.csv", low_memory=False)
    pdf = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted((RAW / "depmap_prism").rglob("*treatment-info.csv"))], ignore_index=True)
    with zipfile.ZipFile(RAW / "ctrp/v2_2/CTRPv2.2_2015_pub_CancerDisc_5_1210.zip") as z:
        with z.open("v22.meta.per_compound.txt") as f:
            cdrug = pd.read_csv(f, sep="\t", low_memory=False)
    drug_sets = {"GDSC": values(gdrug, "DRUG_NAME"), "PRISM": values(pdf, "name"), "CTRP": values(cdrug, "cpd_name")}
    summary = {
        "status": "PASS", "drug_response_values_opened": False,
        "breast_model_identifier_counts": {k: len(v) for k, v in line_sets.items()},
        "breast_model_pairwise_overlap": pairwise(line_sets),
        "compound_name_counts": {k: len(v) for k, v in drug_sets.items()},
        "compound_pairwise_exact_normalized_name_overlap": pairwise(drug_sets),
        "warning": "Name overlap is an audit only; final compound mapping must prioritize stable IDs/InChIKey.",
        "next_gate": "stable_identifier_mapping_and_screen_endpoint_direction_lock",
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
