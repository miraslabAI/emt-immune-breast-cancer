from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SURVIVAL_SOURCE = ROOT / "02_data/03_processed/bulk_validation"
SURVIVAL_LOCK = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
RESPONSE_SOURCE = ROOT / "02_data/03_processed/treatment_response/GSE25066/GSE25066_EMP_response_features.tsv.gz"
RESPONSE_LOCK = ROOT / "02_data/02_interim/treatment_response/locked_manifests"
OUT_DIR = ROOT / "04_analysis/03_feature_engineering/clinical_harmonization"

MISSING_TOKENS = {"", "na", "nan", "n/a", "unknown", "not available", "not reported", "[not available]"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(MISSING_TOKENS))


def add_profiles(rows: list[dict], cohort: str, frame: pd.DataFrame,
                 mapping: dict[str, str], numerical_candidates: set[str]) -> None:
    for standard, source in mapping.items():
        values = clean(frame[source])
        numeric = pd.to_numeric(values, errors="coerce")
        nonmissing = int(values.notna().sum())
        numeric_fraction = float(numeric.notna().sum() / nonmissing) if nonmissing else 0.0
        counts = values.value_counts(dropna=True, sort=True)
        rows.append({
            "cohort": cohort,
            "standard_field": standard,
            "source_column": source,
            "expected_type": "numeric" if standard in numerical_candidates else "categorical_or_ordinal",
            "rows": int(len(frame)),
            "nonmissing": nonmissing,
            "missing": int(values.isna().sum()),
            "coverage": float(values.notna().mean()),
            "numeric_parse_fraction_among_nonmissing": numeric_fraction,
            "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "numeric_median": float(numeric.median()) if numeric.notna().any() else np.nan,
            "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            "distinct_nonmissing_levels": int(values.nunique(dropna=True)),
            "level_counts_json": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
        })


def locked_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, sep="\t", usecols=["sample_id"])["sample_id"].astype(str))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles: list[dict] = []
    inputs: list[dict] = []

    m_path = SURVIVAL_SOURCE / "METABRIC_scores.tsv.gz"
    m_cols = ["Patient ID", "Age at Diagnosis", "Tumor Size", "Lymph nodes examined positive",
              "Neoplasm Histologic Grade", "Pam50 + Claudin-low subtype"]
    m = pd.read_csv(m_path, sep="\t", usecols=m_cols, low_memory=False)
    m = m[m["Patient ID"].astype(str).isin(locked_ids(SURVIVAL_LOCK / "METABRIC_OS_locked_manifest.tsv"))]
    add_profiles(profiles, "METABRIC", m, {
        "age": "Age at Diagnosis", "tumor_burden": "Tumor Size",
        "node_status": "Lymph nodes examined positive", "grade": "Neoplasm Histologic Grade",
        "pam50": "Pam50 + Claudin-low subtype"}, {"age", "tumor_burden", "node_status", "grade"})
    inputs.append({"cohort": "METABRIC", "path": str(m_path), "sha256": sha256(m_path)})

    t_path = SURVIVAL_SOURCE / "TCGA_BRCA_scores.tsv.gz"
    t = pd.read_csv(t_path, sep="\t", usecols=["patient_id", "age_at_diagnosis", "pam50"], low_memory=False)
    t = t[t["patient_id"].astype(str).isin(locked_ids(SURVIVAL_LOCK / "TCGA_BRCA_OS_locked_manifest.tsv"))]
    tcga_clinical_path = ROOT / "02_data/01_raw/bulk_rnaseq/tcga_brca_xena/clinical/BRCA_clinicalMatrix"
    raw = pd.read_csv(tcga_clinical_path, sep="\t", usecols=["sampleID", "pathologic_T", "pathologic_N"], low_memory=False)
    raw["patient_id"] = raw["sampleID"].astype(str).str[:12]
    raw = raw[raw["sampleID"].astype(str).str.contains(r"-01", regex=True)].drop_duplicates("patient_id")
    t = t.merge(raw[["patient_id", "pathologic_T", "pathologic_N"]], on="patient_id", how="left", validate="one_to_one")
    add_profiles(profiles, "TCGA_BRCA", t, {
        "age": "age_at_diagnosis", "tumor_burden": "pathologic_T",
        "node_status": "pathologic_N", "pam50": "pam50"}, {"age"})
    inputs.extend([{"cohort": "TCGA_BRCA", "path": str(t_path), "sha256": sha256(t_path)},
                   {"cohort": "TCGA_BRCA", "path": str(tcga_clinical_path), "sha256": sha256(tcga_clinical_path)}])

    s_path = SURVIVAL_SOURCE / "SCANB_GSE96058_scores.tsv.gz"
    s_cols = ["geo_accession", "age at diagnosis", "tumor size", "lymph node status", "nhg", "pam50 subtype"]
    s = pd.read_csv(s_path, sep="\t", usecols=s_cols, low_memory=False)
    s = s[s["geo_accession"].astype(str).isin(locked_ids(SURVIVAL_LOCK / "SCANB_GSE96058_OS_locked_manifest.tsv"))]
    add_profiles(profiles, "SCANB_GSE96058", s, {
        "age": "age at diagnosis", "tumor_burden": "tumor size", "node_status": "lymph node status",
        "grade": "nhg", "pam50": "pam50 subtype"}, {"age", "tumor_burden", "grade"})
    inputs.append({"cohort": "SCANB_GSE96058", "path": str(s_path), "sha256": sha256(s_path)})

    response_cols = ["geo_accession", "age_years", "clinical_t_stage", "clinical_nodal_status",
                     "grade", "er_status_ihc", "pr_status_ihc", "her2_status", "pam50_class"]
    response = pd.read_csv(RESPONSE_SOURCE, sep="\t", usecols=response_cols, low_memory=False)
    for cohort, manifest in [("GSE25055", "GSE25066_discovery_pCR_manifest.tsv"),
                             ("GSE25065", "GSE25066_validation_IDs_only_manifest.tsv")]:
        d = response[response["geo_accession"].astype(str).isin(locked_ids(RESPONSE_LOCK / manifest))]
        add_profiles(profiles, cohort, d, {
            "age": "age_years", "clinical_T": "clinical_t_stage", "clinical_N": "clinical_nodal_status",
            "grade": "grade", "ER": "er_status_ihc", "PR": "pr_status_ihc",
            "HER2": "her2_status", "pam50": "pam50_class"}, {"age", "grade"})
    inputs.append({"cohort": "GSE25055_GSE25065", "path": str(RESPONSE_SOURCE), "sha256": sha256(RESPONSE_SOURCE)})

    table = pd.DataFrame(profiles).sort_values(["cohort", "standard_field"])
    table_path = OUT_DIR / "clinical_level_profile_outcome_blind.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    summary = {
        "status": "PASS",
        "version": "1.0",
        "outcomes_read": False,
        "outcome_columns_requested_from_sources": [],
        "cohorts": sorted(table["cohort"].unique().tolist()),
        "profile_rows": int(len(table)),
        "inputs": inputs,
        "profile_file": str(table_path),
        "profile_sha256": sha256(table_path),
        "next_gate": "write_and_lock_explicit_clinical_value_dictionaries_and_missing_data_policy",
    }
    summary_path = OUT_DIR / "clinical_level_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

