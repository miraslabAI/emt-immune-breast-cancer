from __future__ import annotations

import os

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SURV_SRC = ROOT / "02_data/03_processed/bulk_validation"
SURV_LOCK = ROOT / "02_data/02_interim/bulk_survival/locked_clinical_manifests"
RESP_SRC = ROOT / "02_data/03_processed/treatment_response/GSE25066/GSE25066_EMP_response_features.tsv.gz"
RESP_LOCK = ROOT / "02_data/02_interim/treatment_response/locked_manifests"
OUT_DIR = ROOT / "02_data/03_processed/analysis_ready_v1/harmonized_clinical_covariates"
REPORT_DIR = ROOT / "04_analysis/03_feature_engineering/clinical_harmonization"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/CLINICAL_HARMONIZATION_LOCK_v1.0.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=["sample_id", "analysis_role"], dtype={"sample_id": str})


def canonical_pam50(value) -> object:
    if pd.isna(value):
        return pd.NA
    key = str(value).strip().lower()
    mapping = {"luma": "LumA", "lumb": "LumB", "basal": "Basal",
               "her2": "HER2-enriched", "normal": "Normal-like",
               "claudin-low": "Other/Claudin-low"}
    return mapping.get(key, pd.NA)


def stage_number(value, prefix: str, permitted: set[int]) -> object:
    if pd.isna(value):
        return pd.NA
    match = re.match(rf"^{prefix}\s*([0-9]+)", str(value).strip(), flags=re.IGNORECASE)
    number = int(match.group(1)) if match else None
    return number if number in permitted else pd.NA


def t_from_mm(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select([x <= 20, (x > 20) & (x <= 50), x > 50], [1, 2, 3], default=np.nan),
                     index=series.index, dtype="Int64")


def join_manifest(manifest: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    if source["sample_id"].duplicated().any():
        raise ValueError("Duplicated source sample IDs")
    out = manifest.merge(source, on="sample_id", how="left", validate="one_to_one", sort=False)
    if len(out) != len(manifest) or out["sample_id"].tolist() != manifest["sample_id"].tolist():
        raise ValueError("Manifest linkage/order failure")
    return out


def write_table(frame: pd.DataFrame, cohort: str, branch: str, outputs: list[dict]) -> None:
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated IDs: {cohort}")
    forbidden = {"os_time_days", "os_event", "pcr_binary", "endpoint"}
    if forbidden.intersection(frame.columns):
        raise ValueError(f"Outcome leakage: {cohort}")
    path = OUT_DIR / f"{cohort}_{branch}_clinical_covariates.tsv"
    frame.to_csv(path, sep="\t", index=False)
    missing = {column: int(frame[column].isna().sum()) for column in frame.columns
               if column not in {"sample_id", "cohort", "analysis_role"}}
    outputs.append({"branch": branch, "cohort": cohort, "rows": int(len(frame)),
                    "columns": frame.columns.tolist(), "missing": missing,
                    "file": str(path), "sha256": sha256(path)})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []

    # METABRIC
    manifest = ordered_manifest(SURV_LOCK / "METABRIC_OS_locked_manifest.tsv")
    cols = ["Patient ID", "Age at Diagnosis", "Tumor Size", "Lymph nodes examined positive",
            "Neoplasm Histologic Grade", "Pam50 + Claudin-low subtype"]
    raw = pd.read_csv(SURV_SRC / "METABRIC_scores.tsv.gz", sep="\t", usecols=cols, low_memory=False)
    raw = raw.rename(columns={"Patient ID": "sample_id"})
    raw["sample_id"] = raw["sample_id"].astype(str)
    d = join_manifest(manifest, raw)
    nodes = pd.to_numeric(d["Lymph nodes examined positive"], errors="coerce")
    out = pd.DataFrame({"sample_id": d.sample_id, "cohort": "METABRIC", "analysis_role": d.analysis_role,
                        "age": pd.to_numeric(d["Age at Diagnosis"], errors="coerce"),
                        "tumor_T": t_from_mm(d["Tumor Size"]),
                        "node_positive": pd.Series(np.where(nodes.notna(), (nodes > 0).astype(float), np.nan), dtype="Int64"),
                        "pam50": d["Pam50 + Claudin-low subtype"].map(canonical_pam50).astype("string"),
                        "grade_sensitivity": pd.to_numeric(d["Neoplasm Histologic Grade"], errors="coerce").astype("Int64")})
    out["node_status_missing"] = out.node_positive.isna().astype("int8")
    out["pam50_complete"] = out.pam50.notna()
    write_table(out, "METABRIC", "survival", outputs)

    # TCGA
    manifest = ordered_manifest(SURV_LOCK / "TCGA_BRCA_OS_locked_manifest.tsv")
    raw = pd.read_csv(SURV_SRC / "TCGA_BRCA_scores.tsv.gz", sep="\t",
                      usecols=["patient_id", "age_at_diagnosis", "pam50"], low_memory=False).rename(columns={"patient_id": "sample_id"})
    raw["sample_id"] = raw.sample_id.astype(str)
    clinical_path = ROOT / "02_data/01_raw/bulk_rnaseq/tcga_brca_xena/clinical/BRCA_clinicalMatrix"
    clinical = pd.read_csv(clinical_path, sep="\t", usecols=["sampleID", "pathologic_T", "pathologic_N"], low_memory=False)
    clinical["sample_id"] = clinical.sampleID.astype(str).str[:12]
    clinical = clinical[clinical.sampleID.astype(str).str.contains(r"-01", regex=True)].drop_duplicates("sample_id")
    raw = raw.merge(clinical[["sample_id", "pathologic_T", "pathologic_N"]], on="sample_id", how="left", validate="one_to_one")
    d = join_manifest(manifest, raw)
    tumor = d.pathologic_T.map(lambda x: stage_number(x, "T", {1, 2, 3, 4})).astype("Int64")
    node_stage = d.pathologic_N.map(lambda x: stage_number(x, "N", {0, 1, 2, 3})).astype("Int64")
    out = pd.DataFrame({"sample_id": d.sample_id, "cohort": "TCGA_BRCA", "analysis_role": d.analysis_role,
                        "age": pd.to_numeric(d.age_at_diagnosis, errors="coerce"), "tumor_T": tumor,
                        "node_positive": node_stage.map(lambda x: pd.NA if pd.isna(x) else int(x > 0)).astype("Int64"),
                        "pam50": d.pam50.map(canonical_pam50).astype("string"),
                        "grade_sensitivity": pd.Series(pd.NA, index=d.index, dtype="Int64")})
    out["node_status_missing"] = out.node_positive.isna().astype("int8")
    out["pam50_complete"] = out.pam50.notna()
    write_table(out, "TCGA_BRCA", "survival", outputs)

    # SCAN-B
    manifest = ordered_manifest(SURV_LOCK / "SCANB_GSE96058_OS_locked_manifest.tsv")
    cols = ["geo_accession", "age at diagnosis", "tumor size", "lymph node status", "nhg", "pam50 subtype"]
    raw = pd.read_csv(SURV_SRC / "SCANB_GSE96058_scores.tsv.gz", sep="\t", usecols=cols, low_memory=False).rename(columns={"geo_accession": "sample_id"})
    raw["sample_id"] = raw.sample_id.astype(str)
    d = join_manifest(manifest, raw)
    node_map = {"NodeNegative": 0, "NodePositive": 1}
    out = pd.DataFrame({"sample_id": d.sample_id, "cohort": "SCANB_GSE96058", "analysis_role": d.analysis_role,
                        "age": pd.to_numeric(d["age at diagnosis"], errors="coerce"),
                        "tumor_T": t_from_mm(d["tumor size"]),
                        "node_positive": d["lymph node status"].map(node_map).astype("Int64"),
                        "pam50": d["pam50 subtype"].map(canonical_pam50).astype("string"),
                        "grade_sensitivity": d.nhg.astype("string").str.extract(r"([123])", expand=False).astype("Int64")})
    out["node_status_missing"] = out.node_positive.isna().astype("int8")
    out["pam50_complete"] = out.pam50.notna()
    write_table(out, "SCANB_GSE96058", "survival", outputs)

    # Response covariates; endpoint columns are never requested.
    response_cols = ["geo_accession", "age_years", "clinical_t_stage", "clinical_nodal_status", "grade",
                     "er_status_ihc", "pr_status_ihc", "her2_status", "pam50_class"]
    response = pd.read_csv(RESP_SRC, sep="\t", usecols=response_cols, low_memory=False).rename(columns={"geo_accession": "sample_id"})
    response["sample_id"] = response.sample_id.astype(str)
    binary_map = {"N": 0, "P": 1}
    for cohort, manifest_name in [("GSE25055", "GSE25066_discovery_pCR_manifest.tsv"),
                                  ("GSE25065", "GSE25066_validation_IDs_only_manifest.tsv")]:
        d = join_manifest(ordered_manifest(RESP_LOCK / manifest_name), response)
        grade = pd.to_numeric(d.grade, errors="coerce").where(lambda x: x.isin([1, 2, 3])).astype("Int64")
        out = pd.DataFrame({"sample_id": d.sample_id, "cohort": cohort, "analysis_role": d.analysis_role,
                            "age": pd.to_numeric(d.age_years, errors="coerce"),
                            "clinical_T": d.clinical_t_stage.map(lambda x: stage_number(x, "T", {0, 1, 2, 3, 4})).astype("Int64"),
                            "clinical_N": d.clinical_nodal_status.map(lambda x: stage_number(x, "N", {0, 1, 2, 3})).astype("Int64"),
                            "grade": grade, "ER": d.er_status_ihc.map(binary_map).astype("Int64"),
                            "PR": d.pr_status_ihc.map(binary_map).astype("Int64"),
                            "HER2": d.her2_status.map(binary_map).astype("Int64"),
                            "pam50": d.pam50_class.map(canonical_pam50).astype("string")})
        out["grade_missing"] = out.grade.isna().astype("int8")
        out["HER2_missing"] = out.HER2.isna().astype("int8")
        out["pam50_complete"] = out.pam50.notna()
        write_table(out, cohort, "response", outputs)

    expected = {"METABRIC": 1979, "TCGA_BRCA": 1080, "SCANB_GSE96058": 3273, "GSE25055": 306, "GSE25065": 182}
    failures = [f"{x['cohort']}: rows={x['rows']} expected={expected[x['cohort']]}" for x in outputs if x["rows"] != expected[x["cohort"]]]
    summary = {"status": "FAIL" if failures else "PASS", "version": "1.0", "outcomes_read": False,
               "imputation_performed": False, "lock_file": str(LOCK), "lock_sha256": sha256(LOCK),
               "predeclared_missing_indicators": {"survival": ["node_status_missing"], "response": ["grade_missing", "HER2_missing"]},
               "outputs": outputs, "failures": failures,
               "next_gate": "independent_covariate_validation_and_analysis_table_assembly"}
    path = REPORT_DIR / "harmonized_clinical_covariate_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

