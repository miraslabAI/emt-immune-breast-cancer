"""Build platform-specific breast cancer model/drug coverage manifests; no modeling."""
from __future__ import annotations

import os

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DRUG = ROOT / "02_data/01_raw/drug_sensitivity"
DEST = ROOT / "02_data/02_interim/pharmacogenomics/coverage_manifests"
REPORT = ROOT / "04_analysis/01_data_audit/pharmacogenomics/coverage_summary.json"
BREAST = re.compile(r"breast|mammary", re.I)


def breast_mask(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    semantic = [c for c in df.columns if re.search(r"lineage|tissue|disease|cancer|site|origin|type", str(c), re.I)]
    if not semantic:
        return pd.Series(False, index=df.index), []
    text = df[semantic].astype("string").fillna("").agg(" | ".join, axis=1)
    return text.str.contains(BREAST, na=False), semantic


def csv_profile(path: Path, label: str) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    mask, fields = breast_mask(df)
    breast = df.loc[mask].copy()
    return {"asset": label, "rows": int(len(df)), "columns": list(map(str, df.columns)),
            "breast_rows": int(mask.sum()), "breast_detection_fields": fields}, breast


def ctrp_v22() -> dict:
    archive = DRUG / "ctrp/v2_2/CTRPv2.2_2015_pub_CancerDisc_5_1210.zip"
    with zipfile.ZipFile(archive) as z:
        with z.open("v22.meta.per_cell_line.txt") as f:
            cells = pd.read_csv(f, sep="\t", low_memory=False)
        with z.open("v22.meta.per_compound.txt") as f:
            compounds = pd.read_csv(f, sep="\t", low_memory=False)
        with z.open("v22.data.auc_sensitivities.txt") as f:
            auc = pd.read_csv(f, sep="\t", low_memory=False)
    mask, fields = breast_mask(cells)
    breast = cells.loc[mask].copy()
    breast.to_csv(DEST / "CTRPv2_2_breast_cell_lines.tsv", sep="\t", index=False)
    return {"cell_lines": int(len(cells)), "breast_cell_lines": int(len(breast)),
            "breast_detection_fields": fields, "compounds": int(len(compounds)),
            "auc_rows": int(len(auc)), "cell_columns": list(cells.columns),
            "compound_columns": list(compounds.columns), "auc_columns": list(auc.columns)}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    assets = {}
    p, breast = csv_profile(DRUG / "depmap_ccle_24Q4/Model.csv", "DepMap_CCLE_Model")
    assets["DepMap_CCLE"] = p
    breast.to_csv(DEST / "DepMap_CCLE_breast_models.tsv", sep="\t", index=False)

    p, breast = csv_profile(DRUG / "gdsc/annotation/model_list_20260921.csv", "GDSC_model_list")
    assets["GDSC_models"] = p
    breast.to_csv(DEST / "GDSC_breast_models.tsv", sep="\t", index=False)
    gdsc_drugs = pd.read_csv(DRUG / "gdsc/annotation/screened_compounds_rel_8.5_inchi.csv", low_memory=False)
    assets["GDSC_compounds"] = {"rows": int(len(gdsc_drugs)), "columns": list(gdsc_drugs.columns)}

    prism_root = DRUG / "depmap_prism"
    prism = []
    for path in sorted(prism_root.rglob("*cell-line-info.csv")):
        p, breast = csv_profile(path, path.stem)
        out = DEST / f"{path.stem}_breast_models.tsv"
        breast.to_csv(out, sep="\t", index=False)
        p["manifest"] = str(out)
        prism.append(p)
    treatment = []
    for path in sorted(prism_root.rglob("*treatment-info.csv")):
        df = pd.read_csv(path, low_memory=False)
        treatment.append({"file": path.name, "rows": int(len(df)), "columns": list(df.columns)})
    assets["PRISM"] = {"cell_line_tables": prism, "treatment_tables": treatment}
    assets["CTRP_v2_2"] = ctrp_v22()

    summary = {
        "status": "PASS", "modeling_performed": False, "assets": assets,
        "important_correction": "PRISM is stored under depmap_prism and is included here; the prior inventory searched the wrong folder name.",
        "next_gate": "cross_platform_cell_line_and_compound_identifier_harmonization",
    }
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
