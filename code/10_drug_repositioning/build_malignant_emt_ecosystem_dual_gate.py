#!/usr/bin/env python
"""Build a malignant-EMT plus ecosystem-barrier dual gate without outcomes."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW_SC = ROOT / "02_data/02_interim/single_cell/GSE176078/GSE176078_raw_counts.h5ad"
MC_SC = ROOT / "02_data/03_processed/single_cell/GSE176078/GSE176078_malignant_patient_metacells.h5ad"
MC_META = ROOT / "02_data/03_processed/single_cell/GSE176078/GSE176078_metacell_metadata_scores.tsv"
TAN = ROOT / "02_data/01_raw/reference/tan2014_emt/Tan2014_generic_EMT_cellline.tsv"
EXPR = ROOT / "02_data/03_processed/analysis_ready_v1/bulk_survival"
FEATURES = ROOT / "04_analysis/11_drug_repositioning/20_therapy_prioritization_feature_lock"
OLD_SCORES = ROOT / "04_analysis/11_drug_repositioning/21_state_specific_therapy_scores/sample_level_therapy_hypothesis_scores.tsv.gz"
OUT = ROOT / "04_analysis/11_drug_repositioning/25_malignant_emt_ecosystem_dual_gate"
COHORTS = ["TCGA_BRCA", "METABRIC", "SCANB_GSE96058"]
EPS = 1e-6


def decode(values):
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]


def categorical(obs, column):
    node = obs[column]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][...]
        categories = np.asarray(decode(node["categories"][...]), dtype=object)
        values = np.full(len(codes), "<MISSING>", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return np.asarray(decode(node[...]), dtype=str)


def obs_index(obs):
    key = obs.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    return np.asarray(decode(obs[key][...]), dtype=str)


def read_h5ad_selected(path, requested, obs_column=None):
    with h5py.File(path, "r") as h:
        ids = obs_index(h["obs"])
        labels = categorical(h["obs"], obs_column) if obs_column else None
        key = h["var"].attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        genes = [x.upper() for x in decode(h["var"][key][...])]
        lookup = {g: i for i, g in enumerate(genes)}
        measured = sorted(set(requested) & set(lookup))
        ix = [lookup[g] for g in measured]
        node = h["X"]
        if isinstance(node, h5py.Dataset):
            full = sparse.csr_matrix(node[...])
        else:
            enc = node.attrs.get("encoding-type", "")
            if isinstance(enc, bytes): enc = enc.decode()
            shape = tuple(int(x) for x in node.attrs["shape"])
            data, indices, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
            full = (sparse.csr_matrix((data, indices, indptr), shape=shape) if enc == "csr_matrix"
                    else sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr())
        library = np.asarray(full.sum(axis=1)).ravel()
        selected = full[:, ix].tocsr()
    return selected, library, ids, labels, measured


def compartment(celltype):
    x = celltype.lower()
    if "cancer epithelial" in x or "malignant" in x or "tumor" in x: return "malignant"
    if "normal epithelial" in x: return "normal_epithelial"
    if any(t in x for t in ["t-cell", "t cell", "b-cell", "b cell", "myeloid", "mast", "plasma"]): return "immune"
    if any(t in x for t in ["caf", "fibro", "endothelial", "pvl", "perivascular", "stroma"]): return "stromal_vascular"
    return "other"


def logcpm(matrix, library):
    scale = np.divide(1e6, library, out=np.zeros_like(library, dtype=float), where=library > 0)
    return np.log1p(matrix.multiply(scale[:, None]).toarray())


def pct(x):
    return pd.Series(x).rank(method="average", pct=True).clip(EPS, 1.0).to_numpy()


def geomean(columns):
    x = np.column_stack(columns)
    return np.exp(np.log(np.clip(x, EPS, 1.0)).mean(axis=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tan = pd.read_csv(TAN, sep="\t")
    candidates = tan.loc[tan["epimes"].eq("mes"), "symbol"].str.upper().tolist()

    raw, raw_lib, _, celltypes, measured = read_h5ad_selected(RAW_SC, candidates, "celltype_major")
    comp = np.asarray([compartment(x) for x in celltypes])
    origin_rows = []
    for label in sorted(np.unique(comp)):
        idx = np.flatnonzero(comp == label)
        counts = np.asarray(raw[idx].sum(axis=0)).ravel()
        cpm = counts / raw_lib[idx].sum() * 1e6
        det = np.asarray((raw[idx] > 0).mean(axis=0)).ravel()
        for gene, value, fraction in zip(measured, cpm, det):
            origin_rows.append({"gene": gene, "compartment": label, "cpm": value, "detection": fraction})
    origin = pd.DataFrame(origin_rows)
    wide_cpm = origin.pivot(index="gene", columns="compartment", values="cpm").fillna(0)
    wide_det = origin.pivot(index="gene", columns="compartment", values="detection").fillna(0)

    mc, mc_lib, mc_ids, _, mc_genes = read_h5ad_selected(MC_SC, candidates)
    if mc_genes != measured:
        common = sorted(set(mc_genes) & set(measured))
        raise ValueError(f"Single-cell gene order mismatch; common={len(common)}")
    mc_expr = pd.DataFrame(logcpm(mc, mc_lib), index=mc_ids, columns=mc_genes)
    meta = pd.read_csv(MC_META, sep="\t").set_index("metacell_id").loc[mc_ids]
    target = meta["M_Tan2014_cellline"]
    rows = []
    for gene in mc_genes:
        rho = float(spearmanr(mc_expr[gene], target).statistic)
        malignant = float(wide_cpm.loc[gene].get("malignant", 0))
        competing = float(wide_cpm.loc[gene].drop(labels=["malignant"], errors="ignore").max())
        ratio = (malignant + EPS) / (competing + EPS)
        detection = float(wide_det.loc[gene].get("malignant", 0))
        eligible = ratio >= 0.5 and detection >= 0.01 and rho >= 0.20
        rows.append({"gene": gene, "metacell_rho_with_Tan2014_M": rho,
                     "malignant_to_best_other_cpm_ratio": ratio,
                     "malignant_detection_fraction": detection, "eligible": eligible})
    audit = pd.DataFrame(rows).sort_values(["eligible", "metacell_rho_with_Tan2014_M"], ascending=[False, False])
    common_bulk_genes = None
    for cohort in COHORTS:
        with h5py.File(EXPR / f"{cohort}_expression.h5", "r") as h:
            cohort_genes = {g.upper() for g in decode(h["gene_symbol"][...])}
        common_bulk_genes = cohort_genes if common_bulk_genes is None else common_bulk_genes & cohort_genes
    audit["cross_cohort_available"] = audit["gene"].isin(common_bulk_genes)
    eligible_all = audit[audit["eligible"]].copy()
    eligible = audit[audit["eligible"] & audit["cross_cohort_available"]].head(12).copy()
    # The minimum-six rule is a validation gate, not a reason to relax cellular-
    # origin criteria after seeing the result.  A 3--5 gene set is retained only
    # as a feasibility core and must not be called a validated compact selector.
    compact_panel_gate_passed = len(eligible) >= 6
    if len(eligible) < 3:
        audit["selected"] = audit["gene"].isin(eligible["gene"])
        audit.to_csv(OUT / "malignant_emt_candidate_audit.tsv", sep="\t", index=False)
        selected = eligible["gene"].tolist()
        pd.DataFrame({"gene": selected, "source": "Tan2014_cellline_mesenchymal",
                      "selection": "feasibility-only; insufficient portable malignant genes"}).to_csv(
            OUT / "locked_malignant_emt_panel.tsv", sep="\t", index=False)
        summary = {
            "status": "REVIEW",
            "created_at": datetime.now().astimezone().isoformat(),
            "analysis_type": "outcome-blind malignant-EMT compact-panel feasibility gate",
            "candidate_genes": len(candidates),
            "measured_in_scRNA": len(measured),
            "strict_cellular_origin_eligible_genes": eligible_all["gene"].tolist(),
            "cross_cohort_portable_genes": selected,
            "compact_malignant_panel_gate_passed": False,
            "reason": "fewer than three malignant-compatible genes were measurable in all three bulk cohorts",
            "important_change": "do not construct a bulk compact malignant-EMT selector; require malignant-cell-resolved qualification",
            "outcomes_used": False,
            "treatment_response_claimed": False,
            "next_gate": "use malignant single-cell EMT states plus spatial tumor/stromal localization as the dual-gate qualification",
        }
        (OUT / "malignant_emt_ecosystem_dual_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    eligible["selected"] = True
    audit = audit.merge(eligible[["gene", "selected"]], on="gene", how="left")
    audit["selected"] = audit["selected"].eq(True)
    audit.to_csv(OUT / "malignant_emt_candidate_audit.tsv", sep="\t", index=False)
    selected = eligible["gene"].tolist()
    pd.DataFrame({"gene": selected, "source": "Tan2014_cellline_mesenchymal",
                  "selection": "GSE176078 malignant-compatible and metacell-correlated"}).to_csv(
        OUT / "locked_malignant_emt_panel.tsv", sep="\t", index=False)

    old = pd.read_csv(OLD_SCORES, sep="\t").set_index(["cohort", "sample_id"])
    validation_rows, enrichment_rows, score_frames = [], [], []
    for cohort in COHORTS:
        with h5py.File(EXPR / f"{cohort}_expression.h5", "r") as h:
            x = h["X"][...].astype(np.float32)
            samples = decode(h["sample_id"][...])
            genes = [g.upper() for g in decode(h["gene_symbol"][...])]
        lookup = {g: i for i, g in enumerate(genes)}
        if not set(selected) <= set(lookup):
            raise ValueError(f"{cohort} missing panel genes: {sorted(set(selected)-set(lookup))}")
        block = x[:, [lookup[g] for g in selected]]
        block = (block - block.mean(axis=0)) / block.std(axis=0, ddof=0)
        malignant_score = block.mean(axis=1)
        f = pd.read_csv(FEATURES / f"{cohort}_therapy_prioritization_features.tsv.gz", sep="\t")
        if f["sample_id"].tolist() != samples:
            raise ValueError(f"Sample order mismatch in {cohort}")
        barrier = np.mean(np.column_stack([pct(f[c]) for c in ["angiogenesis", "hypoxia", "tgfb", "stromal_z"]]), axis=1)
        visibility = np.mean(np.column_stack([pct(f[c]) for c in ["antigen_presentation", "priming_recruitment", "infiltration", "ifng_response"]]), axis=1)
        incomplete = np.mean(np.column_stack([pct(f[c]) for c in ["immune_exclusion_pressure", "suppressive_recruitment"]]), axis=1)
        revised = geomean([pct(malignant_score), barrier, visibility, incomplete])
        original = old.loc[[(cohort, s) for s in samples], "score__mesenchymal_antiangiogenic_IO"].to_numpy()
        rho_full = float(spearmanr(revised, original).statistic)
        top = revised >= np.quantile(revised, 0.90)
        old_top = original >= np.quantile(original, 0.90)
        jaccard = float((top & old_top).sum() / (top | old_top).sum())
        validation_rows.append({"cohort": cohort, "n": len(f), "revised_vs_original_spearman": rho_full,
                                "top10_jaccard": jaccard,
                                "malignant_panel_vs_mesenchymal_core_spearman": float(spearmanr(malignant_score, f["mesenchymal_core"]).statistic)})
        for state, n in f.loc[top, "immune_state"].value_counts().items():
            enrichment_rows.append({"cohort": cohort, "immune_state": state, "top10_n": int(n),
                                    "top10_fraction": float(n/top.sum()), "cohort_fraction": float((f["immune_state"]==state).mean()),
                                    "enrichment_ratio": float((n/top.sum())/(f["immune_state"]==state).mean())})
        score_frames.append(pd.DataFrame({"sample_id": samples, "cohort": cohort,
                                         "malignant_emt_panel_score": malignant_score,
                                         "ecosystem_barrier_score": barrier,
                                         "immune_visibility_score": visibility,
                                         "incomplete_completion_score": incomplete,
                                         "revised_dual_gate_score": revised, "revised_top10": top}))

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "dual_gate_cross_cohort_characterization.tsv", sep="\t", index=False)
    pd.DataFrame(enrichment_rows).to_csv(OUT / "dual_gate_immune_state_enrichment.tsv", sep="\t", index=False)
    pd.concat(score_frames).to_csv(OUT / "dual_gate_sample_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    summary = {
        "status": "PASS" if compact_panel_gate_passed else "REVIEW",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "outcome-blind malignant-EMT plus ecosystem-barrier dual gate",
        "candidate_genes": len(candidates), "measured_in_scRNA": len(measured),
        "selected_malignant_emt_genes": selected,
        "compact_malignant_panel_gate_passed": compact_panel_gate_passed,
        "compact_malignant_panel_required_genes": 6,
        "panel_role": "validated compact selector" if compact_panel_gate_passed else "feasibility core only; not a patient-selection assay",
        "selection_criteria": {"malignant_to_best_other_cpm_ratio_min": 0.5,
                               "malignant_detection_min": 0.01, "metacell_rho_min": 0.20, "maximum_genes": 12},
        "important_change": "malignant EMT and stromal/vascular barrier are independent required gates",
        "outcomes_used": False, "treatment_response_claimed": False,
        "next_gate": "review cross-cohort behavior; if compact gate failed, require malignant-cell-resolved spatial/single-cell qualification rather than relaxing origin criteria",
    }
    (OUT / "malignant_emt_ecosystem_dual_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
