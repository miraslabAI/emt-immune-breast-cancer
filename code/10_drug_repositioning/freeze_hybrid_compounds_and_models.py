#!/usr/bin/env python
"""Freeze exact Hybrid experimental compounds and outcome-blind CCLE model rules."""
from __future__ import annotations

import os

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CCLE_MODEL = ROOT / "02_data/01_raw/drug_sensitivity/depmap_ccle_24Q4/Model.csv"
CCLE_EXPR = ROOT / "02_data/01_raw/drug_sensitivity/depmap_ccle_24Q4/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
STATES = ROOT / "04_analysis/11_drug_repositioning/10_pharmacogenomic_validation_lock/frozen_ccle_breast_emt_states.tsv"
QUAL = ROOT / "04_analysis/11_drug_repositioning/26_multiscale_state_therapy_qualification/multiscale_state_therapy_qualification_summary.json"
OUT = ROOT / "04_analysis/11_drug_repositioning/27_hybrid_compound_model_lock"

ERBB = ["EGFR", "ERBB2", "ERBB3", "PTK6"]
AKT = ["PIK3CA", "AKT1", "AKT2", "AKT3"]
MTOR = ["MTOR", "RPTOR", "RICTOR"]
GENES = ERBB + AKT + MTOR


def column_map(path, genes):
    columns = list(pd.read_csv(path, nrows=0).columns)
    mapping = {}
    for gene in genes:
        hits = [x for x in columns if x == gene or x.startswith(f"{gene} (")]
        if len(hits) != 1: raise ValueError(f"CCLE mapping failed for {gene}: {hits}")
        mapping[gene] = hits[0]
    return columns[0], mapping


def zscore(frame):
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qual = json.loads(QUAL.read_text(encoding="utf-8"))
    if qual.get("status") != "PASS": raise RuntimeError("Stage 26 did not pass")

    compounds = [
        [1, "primary_reference", "sapitinib (AZD8931)", "capivasertib (AZD5363)",
         "pan-EGFR/HER2/HER3", "pan-AKT", "direct breast preclinical synergy and regression; same mechanistic pair",
         "research reference; not a clinical recommendation", "PMID:26095475"],
        [2, "primary_translational_class_test", "neratinib", "capivasertib",
         "irreversible pan-HER", "pan-AKT", "tests whether the reference effect transfers to a currently marketed pan-HER/AKT pair",
         "exact pair efficacy and safety are not established; anticipate overlapping diarrhea/rash", "FDA labels; class-analogue inference"],
        [3, "orthogonal_AKT_control", "sapitinib (AZD8931)", "MK-2206",
         "pan-EGFR/HER2/HER3", "allosteric AKT", "orthogonal AKT chemistry and nominal DrugComb state signal",
         "research-only mechanism control", "project DrugComb stage 17"],
        [4, "secondary_mTOR_reference", "sapitinib (AZD8931)", "vistusertib (AZD2014)",
         "pan-EGFR/HER2/HER3", "dual mTORC1/2", "exact nominal DrugComb discovery pair",
         "no corrected synergy; research-only", "project DrugComb stage 17"],
        [5, "secondary_mTOR_class_test", "neratinib", "vistusertib (AZD2014)",
         "irreversible pan-HER", "dual mTORC1/2", "tests pan-HER class transfer for the mTOR hypothesis",
         "exact pair unvalidated; research-only", "class-analogue inference"],
    ]
    columns = ["priority", "role", "agent_A", "agent_B", "target_A", "target_B", "rationale", "claim_limit", "evidence_anchor"]
    pd.DataFrame(compounds, columns=columns).to_csv(OUT / "locked_hybrid_compound_pairs.tsv", sep="\t", index=False)

    states = pd.read_csv(STATES, sep="\t")
    models = pd.read_csv(CCLE_MODEL, low_memory=False)
    id_col, mapping = column_map(CCLE_EXPR, GENES)
    expr = pd.read_csv(CCLE_EXPR, usecols=[id_col] + list(mapping.values()), low_memory=False)
    rename = {v: k for k, v in mapping.items()}
    rename[id_col] = "ModelID"
    expr = expr.rename(columns=rename)
    frame = states.merge(expr, on="ModelID", how="inner", validate="one_to_one")
    meta_cols = ["ModelID", "SourceDetail", "CatalogNumber", "ModelType", "GrowthPattern", "OnboardedMedia"]
    available_meta = [x for x in meta_cols if x in models]
    frame = frame.merge(models[available_meta].drop_duplicates("ModelID"), on="ModelID", how="left", validate="one_to_one")
    frame[GENES] = frame[GENES].apply(pd.to_numeric, errors="coerce")
    if frame[GENES].isna().any().any(): raise ValueError("Missing baseline CCLE expression")
    z = zscore(frame[GENES])
    frame["erbb_expression_proxy"] = z[ERBB].mean(axis=1)
    frame["akt_expression_proxy"] = z[AKT].mean(axis=1)
    frame["mtor_expression_proxy"] = z[MTOR].mean(axis=1)
    frame["hybrid_coactivation"] = frame[["epithelial_core", "mesenchymal_core"]].min(axis=1)

    hybrid = frame[frame["emt_state"].eq("hybrid-peak")].sort_values(
        ["erbb_expression_proxy", "hybrid_coactivation"], ascending=False).copy()
    hybrid["selection_role"] = "hybrid_candidate"
    hybrid["selection_rank"] = np.arange(1, len(hybrid) + 1)
    hybrid["advance_to_dose_matrix"] = hybrid["selection_rank"].le(6)

    hybrid_low = frame[frame["emt_state"].eq("hybrid-peak")].sort_values("erbb_expression_proxy").head(3).copy()
    hybrid_low["selection_role"] = "hybrid_ERBB_low_control"
    hybrid_low["selection_rank"] = np.arange(1, len(hybrid_low) + 1)
    hybrid_low["advance_to_dose_matrix"] = False
    nonhybrid = frame[~frame["emt_state"].eq("hybrid-peak")].sort_values("erbb_expression_proxy", ascending=False).head(4).copy()
    nonhybrid["selection_role"] = "nonhybrid_ERBB_high_control"
    nonhybrid["selection_rank"] = np.arange(1, len(nonhybrid) + 1)
    nonhybrid["advance_to_dose_matrix"] = False
    selected = pd.concat([hybrid, hybrid_low, nonhybrid], ignore_index=True).drop_duplicates(["ModelID", "selection_role"])
    keep = ["selection_role", "selection_rank", "advance_to_dose_matrix", "ModelID", "StrippedCellLineName",
            "OncotreeSubtype", "ModelSubtypeFeatures", "emt_state", "epithelial_core", "mesenchymal_core",
            "hybrid_coactivation", "erbb_expression_proxy", "akt_expression_proxy", "mtor_expression_proxy",
            *GENES, *[x for x in available_meta if x != "ModelID"]]
    selected[keep].to_csv(OUT / "locked_hybrid_model_shortlist.tsv", sep="\t", index=False)

    rules = [
        [1, "identity", "hybrid-peak by frozen CCLE EMT state; because lines are clonal models, E/M coexpression must still be confirmed by IF or flow"],
        [2, "ERBB qualification", "baseline pEGFR/pHER2/pHER3 or ligand-stimulated ERBB signaling above assay-defined negative control"],
        [3, "AKT qualification", "baseline or inducible pAKT/pPRAS40 plus suppression by capivasertib at a non-saturating exposure"],
        [4, "mTOR qualification", "pS6 and p4EBP1 measurable and suppressible by vistusertib"],
        [5, "controls", "include Hybrid/ERBB-low and non-Hybrid/ERBB-high models to distinguish state specificity from ERBB dependence"],
        [6, "minimum progression", "advance at least four qualified Hybrid models spanning HER2+, TNBC and ER+ when feasible; do not substitute technical replicates"],
    ]
    pd.DataFrame(rules, columns=["order", "gate", "rule"]).to_csv(OUT / "model_qualification_rules.tsv", sep="\t", index=False)

    design = [
        ["calibration", "each single agent", "10-point log dilution", "72-120h", "IC10/IC20/IC50 and maximal effect", "model-specific non-saturating range"],
        ["primary_matrix", "sapitinib or neratinib + capivasertib", "6x6 including zero and 0.25/0.5/1/2x model-specific IC50", "72-120h", "Bliss; Loewe; ZIP; HSA", "predefine viability as primary"],
        ["mechanism", "same primary pairs", "selected matrix points", "2h;8h;24h", "pERBB/pAKT/pPRAS40/pS6/p4EBP1/cleaved-caspase-3", "show feedback blockade and target engagement"],
        ["schedule", "capivasertib lead-in versus simultaneous", "top two qualified models only", "8h lead-in then combination", "feedback-sensitive interaction", "expand only if primary matrix is positive"],
        ["secondary", "mTOR and MK-2206 pairs", "6x6", "72-120h", "same metrics", "run after primary pair gate"],
    ]
    pd.DataFrame(design, columns=["stage", "intervention", "design", "time", "readouts", "frozen_note"]).to_csv(
        OUT / "hybrid_dose_matrix_design.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS", "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind exact-compound and baseline-model lock for Hybrid combination testing",
        "primary_reference_pair": "sapitinib + capivasertib",
        "primary_translational_analogue": "neratinib + capivasertib",
        "orthogonal_AKT_control": "sapitinib + MK-2206",
        "secondary_mTOR_pair": "sapitinib + vistusertib",
        "hybrid_models_ranked": int(len(hybrid)), "hybrid_models_initially_advanced": int(hybrid["advance_to_dose_matrix"].sum()),
        "response_data_used_for_model_selection": False,
        "expression_activity_caveat": "RNA proxies rank candidates only; phosphoprotein gates determine experimental qualification",
        "clinical_recommendation": False,
        "next_gate": "inspect model shortlist, confirm practical availability, then freeze primary/backup models and plate-level analysis plan",
    }
    (OUT / "hybrid_compound_model_lock_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
