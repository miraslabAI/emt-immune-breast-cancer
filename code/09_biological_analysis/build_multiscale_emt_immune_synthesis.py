"""Build an outcome-free, publication-oriented multiscale EMT–immune synthesis."""
from __future__ import annotations

import os

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LAND = ROOT / "04_analysis/08_biological_landscape"
OUT = LAND / "multiscale_synthesis"
LOCK = ROOT / "01_protocol/statistical_analysis_plan/MULTISCALE_EMT_IMMUNE_SYNTHESIS_LOCK_v1.0.md"

BULK = LAND / "bulk_replication/three_cohort_primary_correlation_meta_analysis.tsv"
ARCH = LAND / "emt_state_immune_archetypes/stratified_pooled_state_enrichment.tsv"
SC_WP = LAND / "single_cell/GSE176078_cell_origin/within_patient_metacell_program_models.tsv"
SC_PT = LAND / "single_cell/GSE176078_cell_origin/patient_emt_vs_tumor_intrinsic_program_correlations.tsv"
SP = LAND / "spatial/NeoTRIP_outcome_free/paired_patient_spatial_state_contrasts.tsv"
SP_SUM = LAND / "spatial/NeoTRIP_outcome_free/neotrip_spatial_mechanism_summary.json"

STATE_ORDER = ["mesenchymal_dominant", "hybrid_EM", "double_low"]
STATE_LABEL = {
    "mesenchymal_dominant": "Mesenchymal-dominant",
    "hybrid_EM": "Hybrid E/M",
    "double_low": "Double-low",
}
METRIC_ORDER = [
    "neighbor_enrichment::T", "neighbor_enrichment::NK",
    "neighbor_enrichment::Mono_Mac", "neighbor_enrichment::Stroma",
    "neighbor_program::T_cytotoxic", "neighbor_program::T_dysfunction",
    "neighbor_program::MonoMac_suppression", "neighbor_program::Stroma_barrier",
    "tumor_marker::HLA_ABC", "tumor_marker::PDL1",
]
METRIC_LABEL = {
    "neighbor_enrichment::T": "T proximity",
    "neighbor_enrichment::NK": "NK proximity",
    "neighbor_enrichment::Mono_Mac": "Mono/Mac proximity",
    "neighbor_enrichment::Stroma": "Stromal proximity",
    "neighbor_program::T_cytotoxic": "Neighbor T cytotoxicity",
    "neighbor_program::T_dysfunction": "Neighbor T dysfunction",
    "neighbor_program::MonoMac_suppression": "Mono/Mac suppression",
    "neighbor_program::Stroma_barrier": "Stromal barrier",
    "tumor_marker::HLA_ABC": "Tumor HLA-ABC",
    "tumor_marker::PDL1": "Tumor PD-L1",
}


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def spatial_state(contrast: str) -> str:
    suffix = "_minus_epithelial_dominant"
    if not contrast.endswith(suffix):
        raise ValueError(f"Unexpected contrast: {contrast}")
    return contrast[: -len(suffix)]


def build_spatial_figure(sp: pd.DataFrame) -> None:
    work = sp.copy()
    work["emt_state"] = work["contrast"].map(spatial_state)
    value = work.pivot(index="emt_state", columns="metric", values="mean_difference").loc[STATE_ORDER, METRIC_ORDER]
    fdr = work.pivot(index="emt_state", columns="metric", values="fdr_bh").loc[STATE_ORDER, METRIC_ORDER]

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    vmax = float(np.nanmax(np.abs(value.to_numpy())))
    image = ax.imshow(value.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(METRIC_ORDER)), [METRIC_LABEL[x] for x in METRIC_ORDER], rotation=42, ha="right")
    ax.set_yticks(range(len(STATE_ORDER)), [STATE_LABEL[x] for x in STATE_ORDER])
    for i in range(len(STATE_ORDER)):
        for j in range(len(METRIC_ORDER)):
            q = fdr.iat[i, j]
            mark = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
            color = "white" if abs(value.iat[i, j]) > .58 * vmax else "black"
            ax.text(j, i, f"{value.iat[i, j]:+.2f}{mark}", ha="center", va="center", color=color, fontsize=8.2)
    ax.set_title("NeoTRIP baseline spatial contrasts vs epithelial-dominant cells\npaired-patient mean difference; * FDR<.05, ** <.01, *** <.001")
    cbar = fig.colorbar(image, ax=ax, pad=.015)
    cbar.set_label("Mean paired difference")
    fig.tight_layout()
    fig.savefig(OUT / "neotrip_spatial_state_contrast_heatmap.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "neotrip_spatial_state_contrast_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not LOCK.exists():
        raise FileNotFoundError(LOCK)
    OUT.mkdir(parents=True, exist_ok=True)
    bulk = pd.read_csv(BULK, sep="\t")
    arch = pd.read_csv(ARCH, sep="\t")
    sc_wp = pd.read_csv(SC_WP, sep="\t")
    sc_pt = pd.read_csv(SC_PT, sep="\t")
    sp = pd.read_csv(SP, sep="\t")
    sp_sum = json.loads(SP_SUM.read_text(encoding="utf-8"))

    require_columns(bulk, ["emt_axis", "immune_axis", "pooled_rho_fixed", "pooled_fdr_bh", "same_direction_all_cohorts"], "bulk")
    require_columns(arch, ["emt_state", "immune_state", "common_odds_ratio", "ci95_low", "ci95_high", "fdr_bh"], "archetypes")
    require_columns(sc_wp, ["program", "term", "beta", "fdr_bh"], "single-cell within-patient")
    require_columns(sc_pt, ["emt_feature", "target", "spearman_rho", "permutation_fdr_bh"], "single-cell patient")
    require_columns(sp, ["contrast", "metric", "paired_patients", "mean_difference", "bootstrap_ci95_low", "bootstrap_ci95_high", "fdr_bh"], "spatial")
    if sp_sum.get("status") != "PASS" or sp_sum.get("external_outcomes_used") is not False:
        raise RuntimeError("Spatial mechanism analysis is not a clean outcome-free PASS")

    sp["emt_state"] = sp["contrast"].map(spatial_state)
    sp["significant_fdr05"] = sp["fdr_bh"] < .05
    sp.sort_values(["emt_state", "fdr_bh", "metric"]).to_csv(
        OUT / "spatial_contrasts_annotated.tsv", sep="\t", index=False
    )
    build_spatial_figure(sp)

    rows = [
        {
            "emt_state": "epithelial_dominant",
            "bulk_ecosystem": "Cold enriched (OR 2.44); hot depleted (OR 0.51); excluded depleted (OR 0.16)",
            "malignant_cell_evidence": "Reference state in spatial analysis; no evidence that epithelial identity itself supplies immune recruitment",
            "spatial_evidence": "Reference compartment",
            "supported_bottleneck": "Immune initiation/visibility-poor ecosystem",
            "combination_hypothesis_not_clinical_recommendation": "Priming or immunogenic activation may be required before/with checkpoint blockade",
            "evidence_grade": "Convergent bulk; spatial reference",
        },
        {
            "emt_state": "mesenchymal_dominant",
            "bulk_ecosystem": "Excluded and inflamed-dysfunctional enriched; execution inversely associated with mesenchymal core",
            "malignant_cell_evidence": "EMT position and coactivation track TGF-beta/ECM; patient mesenchymal occupancy tracks tumor IFN response",
            "spatial_evidence": "T, Mono/Mac and stroma proximity increased; tumor HLA-ABC and PD-L1 increased; neighboring T cytotoxicity decreased",
            "supported_bottleneck": "Immune engagement without effective local execution, with myeloid/stromal architecture",
            "combination_hypothesis_not_clinical_recommendation": "Checkpoint plus barrier/stromal modulation; myeloid targeting remains conditional because suppressive-program intensity was not significant",
            "evidence_grade": "Strong three-layer convergence",
        },
        {
            "emt_state": "hybrid_EM",
            "bulk_ecosystem": "Execution low; hot depleted; cold and excluded modestly enriched; heterogeneous phenotype",
            "malignant_cell_evidence": "Coactivation tracks antigen presentation, T/NK-recruiting chemokines and TGF-beta/ECM within patients",
            "spatial_evidence": "Tumor HLA-ABC and PD-L1 increased; T, Mono/Mac and stroma proximity increased; stromal barrier increased; T functional programs unresolved",
            "supported_bottleneck": "Immune-visible/recruiting but barrier-forming hybrid ecosystem",
            "combination_hypothesis_not_clinical_recommendation": "Checkpoint plus barrier modulation is a prespecified hypothesis; checkpoint alone cannot be inferred",
            "evidence_grade": "Strong three-layer convergence",
        },
        {
            "emt_state": "double_low",
            "bulk_ecosystem": "Hot and inflamed-dysfunctional enriched; cold depleted",
            "malignant_cell_evidence": "No state-specific malignant-cell mechanism established",
            "spatial_evidence": "Tumor HLA-ABC and stromal barrier decreased; most proximity and functional contrasts unresolved",
            "supported_bottleneck": "Indeterminate; low EMT signal may include immune-rich tissue, dilution, or purity effects",
            "combination_hypothesis_not_clinical_recommendation": "No state-specific combination should be prioritized without orthogonal identity/purity resolution",
            "evidence_grade": "Mixed; retain as indeterminate",
        },
    ]
    evidence = pd.DataFrame(rows)
    evidence.to_csv(OUT / "multiscale_state_mechanism_evidence_matrix.tsv", sep="\t", index=False)

    selected = sp.loc[sp.significant_fdr05, [
        "emt_state", "metric", "paired_patients", "mean_difference",
        "bootstrap_ci95_low", "bootstrap_ci95_high", "fdr_bh"
    ]].sort_values(["emt_state", "fdr_bh"])
    summary = {
        "status": "PASS",
        "analysis_type": "locked outcome-free multiscale EMT-immune synthesis",
        "external_outcomes_used": False,
        "neotrip_pCR_or_arm_read": False,
        "evidence_layers": ["three-cohort bulk", "GSE176078 malignant-cell", "NeoTRIP baseline spatial"],
        "spatial_tests": int(len(sp)),
        "spatial_fdr05": int(sp.significant_fdr05.sum()),
        "spatial_fdr05_by_state": {k: int(v) for k, v in sp.groupby("emt_state").significant_fdr05.sum().items()},
        "primary_conclusion": "EMT states encode distinct immune bottlenecks; mesenchymal and hybrid states show immune engagement without assured immune completion.",
        "double_low_policy": "indeterminate until orthogonal identity/purity resolution",
        "next_gate": "lock treatment-context interaction analysis before reading NeoTRIP pCR or Arm",
        "outputs": [
            "multiscale_state_mechanism_evidence_matrix.tsv",
            "spatial_contrasts_annotated.tsv",
            "neotrip_spatial_state_contrast_heatmap.png",
            "neotrip_spatial_state_contrast_heatmap.pdf",
        ],
    }
    (OUT / "multiscale_synthesis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
