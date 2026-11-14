from __future__ import annotations

import os

import json
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "02_data/03_processed/analysis_ready_v1/harmonized_clinical_covariates"
SURV_LOCK = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
RESP_LOCK = ROOT / "02_data/02_interim/treatment_response/locked_manifests"
OUT = ROOT / "04_analysis/03_feature_engineering/clinical_harmonization/harmonized_clinical_validation_summary.json"
PAM50 = {"LumA", "LumB", "Basal", "HER2-enriched", "Normal-like", "Other/Claudin-low"}
SPECS = [
    ("survival", "METABRIC", SURV_LOCK / "METABRIC_OS_locked_manifest.tsv"),
    ("survival", "TCGA_BRCA", SURV_LOCK / "TCGA_BRCA_OS_locked_manifest.tsv"),
    ("survival", "SCANB_GSE96058", SURV_LOCK / "SCANB_GSE96058_OS_locked_manifest.tsv"),
    ("response", "GSE25055", RESP_LOCK / "GSE25066_discovery_pCR_manifest.tsv"),
    ("response", "GSE25065", RESP_LOCK / "GSE25066_validation_IDs_only_manifest.tsv"),
]


def subset(values: pd.Series, allowed: set) -> bool:
    return set(values.dropna().unique()).issubset(allowed)


def main() -> None:
    checks, failures = [], []
    forbidden = {"os_time_days", "os_event", "pcr_binary", "endpoint"}
    for branch, cohort, manifest_path in SPECS:
        path = DATA / f"{cohort}_{branch}_clinical_covariates.tsv"
        frame = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
        manifest = pd.read_csv(manifest_path, sep="\t", usecols=["sample_id"], dtype={"sample_id": str})
        local = []
        if frame.sample_id.tolist() != manifest.sample_id.tolist():
            local.append("manifest_sample_order")
        if frame.sample_id.duplicated().any():
            local.append("duplicate_sample_id")
        if forbidden.intersection(frame.columns):
            local.append("outcome_column_present")
        if not subset(frame.pam50, PAM50):
            local.append("pam50_levels")
        if not (frame.pam50_complete.astype(bool) == frame.pam50.notna()).all():
            local.append("pam50_complete_indicator")
        if branch == "survival":
            if not subset(frame.tumor_T, {1, 2, 3, 4}): local.append("tumor_T_levels")
            if not subset(frame.node_positive, {0, 1}): local.append("node_levels")
            if not (frame.node_status_missing.astype(bool) == frame.node_positive.isna()).all():
                local.append("node_missing_indicator")
            if cohort == "TCGA_BRCA" and frame.grade_sensitivity.notna().any():
                local.append("tcga_grade_must_be_missing")
        else:
            if not subset(frame.clinical_T, {0, 1, 2, 3, 4}): local.append("clinical_T_levels")
            if not subset(frame.clinical_N, {0, 1, 2, 3}): local.append("clinical_N_levels")
            if not subset(frame.grade, {1, 2, 3}): local.append("grade_levels")
            for column in ["ER", "PR", "HER2"]:
                if not subset(frame[column], {0, 1}): local.append(f"{column}_levels")
            if not (frame.grade_missing.astype(bool) == frame.grade.isna()).all():
                local.append("grade_missing_indicator")
            if not (frame.HER2_missing.astype(bool) == frame.HER2.isna()).all():
                local.append("HER2_missing_indicator")
        if local:
            failures.append({"branch": branch, "cohort": cohort, "failures": local})
        checks.append({"branch": branch, "cohort": cohort, "rows": len(frame),
                       "pam50_complete": int(frame.pam50_complete.sum()),
                       "status": "FAIL" if local else "PASS"})
    result = {"status": "FAIL" if failures else "PASS", "version": "1.0",
              "outcomes_read": False, "checks": checks, "failures": failures,
              "next_gate": "locked_model_input_assembly"}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()

