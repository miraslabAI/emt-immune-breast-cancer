"""Read-only inventory and schema audit for pharmacogenomic resources."""
from __future__ import annotations

import os

import csv
import gzip
import json
import zipfile
from pathlib import Path


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "04_analysis" / "01_data_audit" / "pharmacogenomics"
DRUG = ROOT / "02_data/01_raw/drug_sensitivity"


def text_header(path: Path, delimiter: str = ",") -> dict:
    result = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        return result
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as fh:
            header = next(csv.reader(fh, delimiter=delimiter))
        result.update({"readable": True, "column_count": len(header), "columns": header})
    except Exception as exc:
        result.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def zip_inventory(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        return result
    try:
        with zipfile.ZipFile(path) as z:
            members = [x for x in z.infolist() if not x.is_dir()]
            result.update({"readable": True, "member_count": len(members),
                           "uncompressed_bytes": int(sum(x.file_size for x in members)),
                           "members": [{"name": x.filename, "bytes": x.file_size} for x in members]})
    except Exception as exc:
        result.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def xlsx_inventory(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        return result
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        result["readable"] = True
        result["sheets"] = [{"name": ws.title, "rows": ws.max_row, "columns": ws.max_column} for ws in wb.worksheets]
        wb.close()
    except Exception as exc:
        result.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = {
        "CCLE_Model": text_header(DRUG / "depmap_ccle_24Q4/Model.csv"),
        "GDSC2_dose_response": xlsx_inventory(DRUG / "gdsc/drug_response/GDSC2_fitted_dose_response_27Oct23.xlsx"),
        "CTRP_archives": [zip_inventory(p) for p in sorted((DRUG / "ctrp").rglob("*.zip"))],
    }
    prism_candidates = sorted((DRUG / "prism").rglob("*.csv*")) if (DRUG / "prism").exists() else []
    assets["PRISM_tables"] = [text_header(p, "\t" if ".tsv" in p.name.lower() else ",") for p in prism_candidates]
    flat = [assets["CCLE_Model"], assets["GDSC2_dose_response"], *assets["CTRP_archives"], *assets["PRISM_tables"]]
    failures = [x["path"] for x in flat if not x.get("exists") or not x.get("readable", False)]
    summary = {
        "audit": "pharmacogenomics_inventory_and_schema", "assets": assets,
        "failures": failures, "status": "PASS" if not failures else "REVIEW",
        "modeling_performed": False,
        "next_gate": "breast_line_mapping_drug_identifier_harmonization_and_coverage",
    }
    (OUT / "pharmacogenomics_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "failures": len(failures),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
