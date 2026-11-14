#!/usr/bin/env python
"""Build a scientific inventory of the copied breast-cancer datasets.

This is a read-only audit of data roots. It records dataset roles, modalities,
file counts, sizes, extensions, and obvious transfer problems. It does not read
clinical outcomes or fit any model.
"""

from __future__ import annotations

import os

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROJECT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))


DATASETS = [
    {
        "dataset_id": "TCGA_BRCA",
        "domain": "bulk_prognosis_multiomics",
        "proposed_role": "external_validation_and_multiomic_mechanism",
        "modalities": "bulk_rnaseq;clinical;survival;mutation;cna;methylation",
        "relative_root": "01_raw/bulk_rnaseq/tcga_brca_xena",
    },
    {
        "dataset_id": "METABRIC",
        "domain": "bulk_prognosis",
        "proposed_role": "prognostic_model_development",
        "modalities": "microarray;clinical;survival;mutation;cna;methylation",
        "relative_root": "01_raw/external_validation/metabric/brca_metabric",
    },
    {
        "dataset_id": "GSE96058_SCANB",
        "domain": "bulk_prognosis",
        "proposed_role": "external_validation",
        "modalities": "bulk_rnaseq;clinical;survival",
        "relative_root": "01_raw/external_validation/GSE96058",
    },
    {
        "dataset_id": "GSE25066",
        "domain": "neoadjuvant_response",
        "proposed_role": "GSE25055_discovery_GSE25065_locked_validation",
        "modalities": "microarray;pcr;rcb;drfs;clinical",
        "relative_root": "01_raw/treatment_response/GSE25066",
    },
    {
        "dataset_id": "GSE194040_ISPY2",
        "domain": "neoadjuvant_response",
        "proposed_role": "treatment_transport_validation",
        "modalities": "microarray;treatment;response;clinical",
        "relative_root": "01_raw/treatment_response/GSE194040",
    },
    {
        "dataset_id": "GSE173839_ISPY2_DURVA_OLAPARIB",
        "domain": "neoadjuvant_response",
        "proposed_role": "treatment_specific_secondary_validation",
        "modalities": "microarray;treatment;response;clinical",
        "relative_root": "01_raw/treatment_response/GSE173839",
    },
    {
        "dataset_id": "GSE240671",
        "domain": "paired_treatment_dynamics",
        "proposed_role": "secondary_bulk_longitudinal_validation",
        "modalities": "bulk_rnaseq;paired_pre_post;rcb",
        "relative_root": "01_raw/treatment_response/GSE240671",
    },
    {
        "dataset_id": "INTEGRATED_BRCA_ATLAS_2026",
        "domain": "single_cell",
        "proposed_role": "cellular_origin_and_specificity",
        "modalities": "scrna;malignant_annotations;patient_metadata",
        "relative_root": "01_raw/single_cell/Integrated_BRCA_Atlas_2026",
    },
    {
        "dataset_id": "GSE176078",
        "domain": "single_cell",
        "proposed_role": "single_cell_discovery_support",
        "modalities": "scrna;bulk_rnaseq;metadata",
        "relative_root": "01_raw/single_cell/GSE176078",
    },
    {
        "dataset_id": "GSE161529",
        "domain": "single_cell",
        "proposed_role": "single_cell_replication",
        "modalities": "scrna;metadata",
        "relative_root": "01_raw/single_cell/GSE161529",
    },
    {
        "dataset_id": "GSE225600",
        "domain": "single_cell_spatial",
        "proposed_role": "paired_single_cell_spatial_validation",
        "modalities": "scrna;spatial_transcriptomics;histology",
        "relative_root": "01_raw/single_cell_spatial/GSE225600",
    },
    {
        "dataset_id": "HTAN_SLIDESEQ_2024",
        "domain": "spatial",
        "proposed_role": "spatial_replication",
        "modalities": "slideseq;metadata",
        "relative_root": "01_raw/spatial/HTAN_Klughammer_2024_SlideSeq",
    },
    {
        "dataset_id": "NEOTRIP_QUICHE",
        "domain": "spatial_proteomics",
        "proposed_role": "treatment_response_spatial_validation",
        "modalities": "spatial_proteomics;pcr;treatment_arm",
        "relative_root": "01_raw/spatial/NeoTRIP_QUICHE_14290163",
    },
    {
        "dataset_id": "TONIC",
        "domain": "spatial_longitudinal",
        "proposed_role": "treatment_niche_dynamics",
        "modalities": "spatial_proteomics;longitudinal;clinical",
        "relative_root": "01_raw/spatial/TONIC_14112853",
    },
    {
        "dataset_id": "DEPMAP_CCLE_24Q4",
        "domain": "functional_genomics",
        "proposed_role": "cell_line_expression_and_dependency_context",
        "modalities": "expression;model_metadata",
        "relative_root": "01_raw/drug_sensitivity/depmap_ccle_24Q4",
    },
    {
        "dataset_id": "GDSC",
        "domain": "drug_response",
        "proposed_role": "drug_discovery",
        "modalities": "expression;dose_response;compound_metadata",
        "relative_root": "01_raw/drug_sensitivity/gdsc",
    },
    {
        "dataset_id": "CTRP",
        "domain": "drug_response",
        "proposed_role": "drug_replication",
        "modalities": "dose_response;compound_metadata;model_metadata",
        "relative_root": "01_raw/drug_sensitivity/ctrp",
    },
    {
        "dataset_id": "PRISM",
        "domain": "drug_response",
        "proposed_role": "drug_replication",
        "modalities": "viability;compound_metadata;model_metadata",
        "relative_root": "01_raw/drug_sensitivity/depmap_prism",
    },
    {
        "dataset_id": "LINCS_GSE70138",
        "domain": "perturbational_transcriptomics",
        "proposed_role": "mechanistic_direction_only",
        "modalities": "level5_expression;compound_metadata;cell_metadata",
        "relative_root": "01_raw/reference/lincs/GSE70138",
    },
]


SUSPICIOUS_SUFFIXES = {".part", ".partial", ".tmp", ".crdownload", ".download"}


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    domain: str
    proposed_role: str
    modalities: str
    relative_root: str
    root_exists: bool
    file_count: int
    total_bytes: int
    zero_byte_files: int
    suspicious_partial_files: int
    extension_counts: str
    summary_json_count: int
    manifest_file_count: int
    audit_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cohort inventory.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def normalized_extension(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in ([".tsv", ".gz"], [".csv", ".gz"], [".txt", ".gz"], [".mtx", ".gz"]):
        return "".join(suffixes[-2:])
    return path.suffix.lower() or "[none]"


def inspect_dataset(data_root: Path, specification: dict) -> DatasetRecord:
    relative_root = specification["relative_root"]
    root = data_root / Path(relative_root)
    if not root.exists():
        return DatasetRecord(
            **specification,
            root_exists=False,
            file_count=0,
            total_bytes=0,
            zero_byte_files=0,
            suspicious_partial_files=0,
            extension_counts="{}",
            summary_json_count=0,
            manifest_file_count=0,
            audit_status="missing_root",
        )

    files = [path for path in root.rglob("*") if path.is_file()]
    extension_counts = Counter(normalized_extension(path) for path in files)
    zero_byte = sum(path.stat().st_size == 0 for path in files)
    suspicious = sum(
        path.suffix.lower() in SUSPICIOUS_SUFFIXES
        or any(token in path.name.lower() for token in ("partial", "incomplete"))
        for path in files
    )
    summary_json_count = sum(
        path.suffix.lower() == ".json" and "summary" in path.name.lower()
        for path in files
    )
    manifest_count = sum("manifest" in path.name.lower() for path in files)

    if not files:
        status = "empty_root"
    elif zero_byte:
        status = "review_zero_byte_files"
    elif suspicious:
        status = "review_partial_files"
    else:
        status = "ready_for_cohort_specific_audit"

    return DatasetRecord(
        **specification,
        root_exists=True,
        file_count=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
        zero_byte_files=zero_byte,
        suspicious_partial_files=suspicious,
        extension_counts=json.dumps(dict(sorted(extension_counts.items()))),
        summary_json_count=summary_json_count,
        manifest_file_count=manifest_count,
        audit_status=status,
    )


def write_tsv(path: Path, rows: list[DatasetRecord]) -> None:
    fields = list(DatasetRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "02_data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    rows = [inspect_dataset(data_root, specification) for specification in DATASETS]
    manifest_dir = data_root / "02_metadata" / "sample_manifests"
    report_dir = project_root / "04_analysis" / "01_data_audit"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = manifest_dir / "cohort_inventory.tsv"
    summary_path = report_dir / "cohort_inventory_summary.json"
    write_tsv(inventory_path, rows)

    status_counts = Counter(row.audit_status for row in rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "data_root": str(data_root),
        "dataset_count": len(rows),
        "present_dataset_roots": sum(row.root_exists for row in rows),
        "total_registered_files": sum(row.file_count for row in rows),
        "total_registered_bytes": sum(row.total_bytes for row in rows),
        "zero_byte_files": sum(row.zero_byte_files for row in rows),
        "suspicious_partial_files": sum(row.suspicious_partial_files for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "datasets_requiring_review": [
            row.dataset_id
            for row in rows
            if row.audit_status != "ready_for_cohort_specific_audit"
        ],
        "inventory_tsv": str(inventory_path),
        "next_gate": "cohort_specific_schema_and_endpoint_audit",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Inventory: {inventory_path}")
    print(f"Summary: {summary_path}")
    return 0 if not summary["datasets_requiring_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
