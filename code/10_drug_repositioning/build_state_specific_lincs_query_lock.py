#!/usr/bin/env python
"""Freeze state-specific LINCS objectives before opening Level-5 values.

This stage is deliberately metadata-only.  It creates auditable pathway/gene
modules, preservation constraints, and the eligible breast-cell perturbation
universe.  It never reads the LINCS Level-5 response matrix.
"""
from __future__ import annotations

import os

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
REPRESENTATIVES = ROOT / "04_analysis/11_drug_repositioning/06_collapsed_state_pathway_map/state_pathway_representatives_annotated.tsv"
TARGETS = ROOT / "04_analysis/11_drug_repositioning/07_state_druggable_target_map/state_druggable_target_evidence.tsv"
DRUGS = ROOT / "04_analysis/11_drug_repositioning/07_state_druggable_target_map/state_chembl_drug_mechanism_candidates.tsv.gz"
LINCS_DIR = ROOT / "02_data/01_raw/reference/lincs/GSE70138"
GENE_INFO = LINCS_DIR / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
CELL_INFO = LINCS_DIR / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
PERT_INFO = LINCS_DIR / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz"
SIG_INFO = LINCS_DIR / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
SIG_METRICS = LINCS_DIR / "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz"
LEVEL5 = LINCS_DIR / "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/EMT_IMMUNE_DRUG_REPOSITIONING_LOCK_v1.0.md"
OUT = ROOT / "04_analysis/11_drug_repositioning/08_state_specific_lincs_query_lock"

MIN_MODULE_BING_GENES = 5
MIN_DISTIL_REPLICATES = 2
MIN_TAS = 0.20

# These are preservation constraints, not reversal targets.  A compound that
# suppresses them is penalized even if it reverses a barrier program.
PRESERVE_MODULES = {
    "malignant_antigen_visibility": {
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "NLRC5", "PSMB8", "PSMB9", "PSMB10", "ERAP1", "ERAP2", "CALR",
        "CANX", "PDIA3",
    },
    "malignant_ifn_response": {
        "STAT1", "STAT2", "IRF1", "IRF7", "IRF9", "JAK1", "JAK2",
        "IFNGR1", "IFNGR2", "ISG15", "MX1", "OAS1", "OAS2", "OAS3",
        "GBP1", "GBP2", "IFIH1", "DDX58",
    },
    "t_nk_recruitment_support": {
        "CXCL9", "CXCL10", "CXCL11", "CCL4", "CCL5", "XCL1", "XCL2",
        "FLT3LG", "IL15",
    },
}

# Pathways are selected from the already frozen multiscale catalogue.  The
# lists distinguish actionable barrier/driver hypotheses from mixed immune
# signals that must not be naively reversed.
EXACT_OBJECTIVES = {
    "hybrid-peak": {
        "hybrid_malignant_plasticity_barrier": {
            "Regulation of MITF-M-dependent genes involved in extracellular matrix, focal adhesion and epithelial-to-mesenchymal transition",
            "EPH-ephrin mediated repulsion of cells",
            "Type I hemidesmosome assembly",
            "Gap junction trafficking and regulation",
            "RHO GTPases activate IQGAPs",
            "Sema4D in semaphorin signaling",
            "EPHA-mediated growth cone collapse",
            "Apoptotic cleavage of cell adhesion  proteins",
        },
        "hybrid_growth_signal": {
            "ERBB2 Activates PTK6 Signaling",
            "TFAP2 (AP-2) family regulates transcription of growth factors and their receptors",
        },
    },
    "mesenchymal-enriched": {
        "mesenchymal_ecm_integrin_barrier": {
            "Molecules associated with elastic fibres",
            "Integrin cell surface interactions",
            "Platelet Adhesion to exposed collagen",
            "Cell-extracellular matrix interactions",
            "Other semaphorin interactions",
            "RHO GTPases Activate NADPH Oxidases",
            "NCAM1 interactions",
            "Ephrin signaling",
            "Basigin interactions",
        },
        "mesenchymal_hypoxia_angiogenesis": {
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_HYPOXIA",
        },
        "mesenchymal_coagulation_platelet": {
            "HALLMARK_COAGULATION",
            "Diseases of hemostasis",
            "RUNX1 regulates genes involved in megakaryocyte differentiation and platelet function",
        },
        "mesenchymal_protumor_inflammation": {
            "Interleukin-4 and Interleukin-13 signaling",
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING",
            "Non-canonical inflammasome activation",
            "Cell recruitment (pro-inflammatory response)",
        },
        "mesenchymal_growth_escape": {
            "HALLMARK_KRAS_SIGNALING_UP",
            "HALLMARK_HEDGEHOG_SIGNALING",
            "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def numeric_token(value: object) -> float:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group()) if match else np.nan


def distil_count(value: object) -> int:
    text = str(value)
    return 0 if not text or text.casefold() == "nan" else len(text.split("|"))


def add_module_rows(rows: list[dict], state: str, module: str, role: str,
                    action: str, genes: set[str], source_pathways: list[str],
                    bing: set[str], landmark: set[str], evidence: str,
                    compartment: str) -> None:
    for gene in sorted(genes):
        rows.append({
            "state": state,
            "module": module,
            "module_role": role,
            "desired_lincs_direction": action,
            "gene": gene,
            "lincs_bing": gene in bing,
            "lincs_landmark": gene in landmark,
            "source_pathways": ";".join(source_pathways),
            "evidence_tier": evidence,
            "intended_compartment": compartment,
        })


def main() -> None:
    required = [REPRESENTATIVES, TARGETS, DRUGS, GENE_INFO, CELL_INFO, PERT_INFO,
                SIG_INFO, SIG_METRICS, LEVEL5, LOCK]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    representatives = pd.read_csv(REPRESENTATIVES, sep="\t", low_memory=False)
    representatives["detected_gene_symbols"] = representatives["detected_gene_symbols"].fillna("")
    lincs_genes = pd.read_csv(GENE_INFO, sep="\t", compression="gzip", dtype=str)
    lincs_genes["gene"] = lincs_genes["pr_gene_symbol"].astype(str).str.upper()
    bing = set(lincs_genes.loc[pd.to_numeric(lincs_genes["pr_is_bing"], errors="coerce").eq(1), "gene"])
    landmark = set(lincs_genes.loc[pd.to_numeric(lincs_genes["pr_is_lm"], errors="coerce").eq(1), "gene"])

    failures: list[str] = []
    module_rows: list[dict] = []

    # Epithelial state: LINCS is exploratory for growth/proliferation only.
    # Metabolic programs are explicitly deferred to viability/dependency data.
    epithelial = representatives[
        representatives["bulk_state_class"].eq("epithelial-enriched")
        & representatives["biological_theme"].isin(["growth_survival_signaling", "cell_cycle_dna_damage"])
    ].copy()
    epithelial_genes = set()
    for value in epithelial["detected_gene_symbols"]:
        epithelial_genes.update(g for g in str(value).split(";") if g)
    add_module_rows(
        module_rows, "epithelial-enriched", "epithelial_growth_proliferation",
        "exploratory_target", "decrease", epithelial_genes,
        sorted(epithelial["pathway"].astype(str)), bing, landmark,
        "bulk_multicohort;requires_PRISM_GDSC_dependency", "malignant",
    )

    # Hybrid and mesenchymal objectives are exact, frozen pathway collections.
    observed_pathways = set(representatives["pathway"].astype(str))
    for state, modules in EXACT_OBJECTIVES.items():
        for module, requested_pathways in modules.items():
            absent = sorted(requested_pathways - observed_pathways)
            if absent:
                failures.append(f"missing_frozen_pathways:{module}:{'|'.join(absent)}")
            selected = representatives[
                representatives["bulk_state_class"].eq(state)
                & representatives["pathway"].astype(str).isin(requested_pathways)
            ].copy()
            genes: set[str] = set()
            for value in selected["detected_gene_symbols"]:
                genes.update(g for g in str(value).split(";") if g)
            malignant_supported = selected["cell_origin_evidence"].eq("malignant_supported").any()
            compartment = "malignant" if malignant_supported else "ecosystem_context_tested_in_malignant_cells"
            evidence = "multiscale_malignant_supported" if malignant_supported else "bulk_multicohort_context"
            add_module_rows(
                module_rows, state, module, "primary_target", "decrease", genes,
                sorted(selected["pathway"].astype(str)), bing, landmark,
                evidence, compartment,
            )

    # The same preservation constraints are applied independently within each
    # state; they are not interpreted as state-defining pathways.
    for state in ["epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"]:
        for module, genes in PRESERVE_MODULES.items():
            add_module_rows(
                module_rows, state, module, "preservation_constraint",
                "preserve_or_increase", genes, ["curated_immune_visibility_constraint"],
                bing, landmark, "prior_biology_plus_existing_scRNA_spatial_results",
                "malignant" if module != "t_nk_recruitment_support" else "malignant_immune_interface",
            )

    manifest = pd.DataFrame(module_rows).drop_duplicates(["state", "module", "gene"])
    coverage_rows = []
    for keys, group in manifest.groupby(["state", "module", "module_role"], sort=True):
        state, module, role = keys
        n_bing = int(group["lincs_bing"].sum())
        coverage_rows.append({
            "state": state,
            "module": module,
            "module_role": role,
            "desired_lincs_direction": group["desired_lincs_direction"].iloc[0],
            "genes": int(len(group)),
            "lincs_bing_genes": n_bing,
            "lincs_landmark_genes": int(group["lincs_landmark"].sum()),
            "bing_coverage": float(n_bing / len(group)) if len(group) else 0.0,
            "intended_compartment": group["intended_compartment"].iloc[0],
            "evidence_tier": group["evidence_tier"].iloc[0],
        })
        if n_bing < MIN_MODULE_BING_GENES:
            failures.append(f"insufficient_bing_coverage:{state}:{module}:{n_bing}")
    coverage = pd.DataFrame(coverage_rows)

    # Metadata-only LINCS eligibility universe.  No Level-5 values are opened.
    cells = pd.read_csv(CELL_INFO, sep="\t", compression="gzip", low_memory=False)
    breast_cells = set(cells.loc[
        cells["primary_site"].astype("string").str.casefold().eq("breast")
        & cells["sample_type"].astype("string").str.casefold().eq("tumor"),
        "cell_id",
    ].astype(str))
    signatures = pd.read_csv(
        SIG_INFO, sep="\t", compression="gzip",
        usecols=["sig_id", "pert_id", "pert_iname", "pert_type", "cell_id",
                 "pert_idose", "pert_itime", "distil_id"], low_memory=False,
    )
    signatures = signatures[
        signatures["cell_id"].astype(str).isin(breast_cells)
        & signatures["pert_type"].astype(str).str.startswith("trt_cp", na=False)
    ].copy()
    signatures["dose_value"] = signatures["pert_idose"].map(numeric_token)
    signatures["time_h"] = signatures["pert_itime"].map(numeric_token)
    signatures["distil_nsample_from_info"] = signatures["distil_id"].map(distil_count)

    metrics = pd.read_csv(SIG_METRICS, sep="\t", compression="gzip", low_memory=False)
    metrics = metrics[[c for c in ["sig_id", "tas", "distil_nsample", "distil_cc_q75", "distil_ss"] if c in metrics.columns]]
    metrics = metrics.drop_duplicates("sig_id")
    signatures = signatures.merge(metrics, on="sig_id", how="left", validate="one_to_one")
    signatures["tas"] = pd.to_numeric(signatures.get("tas"), errors="coerce")
    signatures["distil_nsample"] = pd.to_numeric(signatures.get("distil_nsample"), errors="coerce")
    replicate_count = signatures["distil_nsample"].fillna(signatures["distil_nsample_from_info"])
    signatures["qc_pass"] = signatures["tas"].ge(MIN_TAS) & replicate_count.ge(MIN_DISTIL_REPLICATES)

    eligible = signatures[signatures["qc_pass"]].copy()
    if eligible.empty:
        failures.append("no_qc_eligible_breast_tumor_compound_signatures")
    if eligible["sig_id"].duplicated().any():
        failures.append("duplicate_eligible_signature_ids")

    perts = pd.read_csv(PERT_INFO, sep="\t", compression="gzip", low_memory=False)
    perts = perts[perts["pert_type"].astype(str).str.startswith("trt_cp", na=False)].copy()
    perts["normalized_name"] = perts["pert_iname"].map(normal_name)
    drugs = pd.read_csv(DRUGS, sep="\t", compression="gzip", low_memory=False)
    drugs = drugs[drugs["mechanism_direction_compatible"].astype(str).str.casefold().eq("true")].copy()
    drugs["normalized_name"] = drugs["molecule_name"].map(normal_name)
    direct_overlap = drugs.merge(
        perts[["pert_id", "pert_iname", "canonical_smiles", "inchi_key", "normalized_name"]],
        on="normalized_name", how="inner", suffixes=("_chembl", "_lincs"),
    )

    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "frozen_state_lincs_gene_modules.tsv", sep="\t", index=False)
    coverage.to_csv(OUT / "frozen_state_lincs_module_coverage.tsv", sep="\t", index=False)
    eligible.to_csv(OUT / "eligible_breast_lincs_signatures.tsv.gz", sep="\t", index=False, compression="gzip")
    direct_overlap.to_csv(OUT / "direction_compatible_chembl_lincs_name_matches.tsv", sep="\t", index=False)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "FAIL" if failures else "PASS",
        "analysis_type": "outcome-free state-specific LINCS query freeze and metadata preflight",
        "lincs_level5_values_loaded_or_used": False,
        "whole_state_reversal_primary": False,
        "one_drug_for_all_states_assumed": False,
        "module_scoring_rule": "decrease harmful state-specific modules while penalizing loss of immune-visibility preservation modules",
        "epithelial_constraint": "metabolic programs require PRISM/GDSC dependency support before inhibition",
        "modules": int(coverage.shape[0]),
        "module_gene_rows": int(manifest.shape[0]),
        "breast_tumor_cell_lines_in_metadata": int(len(breast_cells)),
        "eligible_signatures": int(len(eligible)),
        "eligible_cell_lines": int(eligible["cell_id"].nunique()),
        "eligible_compounds": int(eligible["pert_id"].nunique()),
        "eligible_time_hours": sorted(float(x) for x in eligible["time_h"].dropna().unique()),
        "direction_compatible_chembl_rows": int(len(drugs)),
        "direct_chembl_lincs_name_match_rows": int(len(direct_overlap)),
        "direct_chembl_lincs_name_match_compounds": int(direct_overlap["pert_id"].nunique()) if len(direct_overlap) else 0,
        "qc": {"minimum_tas": MIN_TAS, "minimum_distil_replicates": MIN_DISTIL_REPLICATES},
        "failures": failures,
        "input_sha256": {str(path): sha256(path) for path in required if path != LEVEL5},
        "level5_path_present_but_not_opened": str(LEVEL5),
        "next_gate": "inspect and accept frozen modules/coverage, then score Level-5 signatures in chunks",
    }
    (OUT / "state_specific_lincs_query_lock_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
