#!/usr/bin/env python
"""Generate paper-ready figures, tables, Results modules, and legends from frozen outputs."""
from __future__ import annotations

import os

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ANALYSIS = ROOT / "04_analysis/11_drug_repositioning"
ATLAS_DIR = ANALYSIS / "31_final_computational_evidence_synthesis"
FIG_MAIN = ROOT / "05_results/figures/main"
FIG_SUPP = ROOT / "05_results/figures/supplementary"
TAB_MAIN = ROOT / "05_results/tables/main"
TAB_SUPP = ROOT / "05_results/tables/supplementary"
REPORT_DIR = ROOT / "05_results/reports"
MANUSCRIPT = ROOT / "06_manuscript/main_text"

STATE_COLORS = {
    "epithelial-enriched": "#3B6EA8",
    "hybrid-peak": "#E6A23C",
    "mesenchymal-enriched": "#A94B5A",
}
FORMULA_LABELS = {
    "hybrid_panERBB_AKT": "Hybrid: pan-ErbB + AKT",
    "hybrid_panERBB_mTORC12": "Hybrid: pan-ErbB + mTORC1/2",
    "mesenchymal_antiangiogenic_IO": "Mesenchymal: vascular remodeling + IO",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    outputs = []
    for suffix in [".png", ".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor="white")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def figure_1_state_biology(themes: pd.DataFrame, states: pd.DataFrame) -> list[str]:
    core_themes = [
        "adhesion_ecm_motility", "growth_survival_signaling", "immune_inflammatory",
        "hypoxia_angiogenesis", "metabolic_adaptation", "protein_rna_homeostasis",
        "cell_cycle_dna_damage", "coagulation_platelet",
    ]
    theme_labels = {
        "adhesion_ecm_motility": "Adhesion/ECM/motility",
        "growth_survival_signaling": "Growth/survival",
        "immune_inflammatory": "Immune/inflammatory",
        "hypoxia_angiogenesis": "Hypoxia/angiogenesis",
        "metabolic_adaptation": "Metabolic adaptation",
        "protein_rna_homeostasis": "Protein/RNA homeostasis",
        "cell_cycle_dna_damage": "Cell cycle/DNA damage",
        "coagulation_platelet": "Coagulation/platelet",
    }
    state_order = ["epithelial-enriched", "hybrid-peak", "mesenchymal-enriched"]
    pivot = themes[themes["biological_theme"].isin(core_themes)].pivot_table(
        index="bulk_state_class", columns="biological_theme", values="representative_pathways", fill_value=0
    ).reindex(state_order).reindex(columns=core_themes, fill_value=0)

    fig = plt.figure(figsize=(12.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    palette = ["#758AA8", "#D29A46", "#B65C71", "#8B6BAE", "#5A9B79", "#8E8E8E", "#C87652", "#6C8F95"]
    left = np.zeros(len(pivot))
    for theme, color in zip(core_themes, palette):
        values = pivot[theme].to_numpy()
        ax.barh(np.arange(len(pivot)), values, left=left, color=color, edgecolor="white", linewidth=0.5,
                label=theme_labels[theme])
        left += values
    ax.set_yticks(np.arange(len(state_order)), ["Epithelial-enriched", "Hybrid-peak", "Mesenchymal-enriched"])
    for tick, state in zip(ax.get_yticklabels(), state_order):
        tick.set_color(STATE_COLORS[state])
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Nonredundant pathway representatives")
    ax.set_title("State-specific pathway architecture", loc="left", fontweight="bold")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    panel_label(ax, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    panel_label(ax2, "B")
    ax2.set_title("Therapeutic positioning", loc="left", fontweight="bold", pad=6)
    rows = [
        ("Epithelial-enriched", STATE_COLORS["epithelial-enriched"],
         "Metabolic and proteostatic programs", "No combination promoted", "Resolve dependency vs adaptation"),
        ("Hybrid-peak", STATE_COLORS["hybrid-peak"],
         "ERBB dependency + immune-visible barrier", "pan-ErbB + AKT (primary)", "Independent dose-matrix testing"),
        ("Mesenchymal-enriched", STATE_COLORS["mesenchymal-enriched"],
         "ECM/vascular barrier + immune incompletion", "Vascular remodeling + checkpoint", "Vascularized immune model required"),
    ]
    y = 0.92
    for title, color, biology, therapy, next_step in rows:
        box = FancyBboxPatch((0.02, y - 0.23), 0.96, 0.20, boxstyle="round,pad=0.012,rounding_size=0.015",
                             linewidth=1.2, edgecolor=color, facecolor="white")
        ax2.add_patch(box)
        ax2.text(0.05, y - 0.055, title, color=color, fontweight="bold", fontsize=10, transform=ax2.transAxes)
        ax2.text(0.05, y - 0.105, textwrap.fill(biology, 48), fontsize=8.5, transform=ax2.transAxes)
        ax2.text(0.05, y - 0.155, "Position: " + therapy, fontsize=8.5, fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.05, y - 0.198, "Next: " + next_step, fontsize=8, color="#555555", transform=ax2.transAxes)
        y -= 0.29
    fig.suptitle("EMT states define distinct biological and therapeutic contexts", fontsize=14, fontweight="bold", y=1.02)
    return save_figure(fig, FIG_MAIN / "Figure_1_state_biology_and_therapy_positioning")


def figure_2_multiscale_gates(gates: pd.DataFrame) -> list[str]:
    strategy_order = ["hybrid_panERBB_vertical_blockade", "mesenchymal_antiangiogenic_IO"]
    gate_order = list(dict.fromkeys(gates["gate"].tolist()))
    matrix = pd.DataFrame(index=gate_order, columns=strategy_order, dtype=float)
    label_matrix = pd.DataFrame("", index=gate_order, columns=strategy_order)
    mapping = {"PASS": 2, "FAIL": 0, "UNRESOLVED": 1}
    for row in gates.itertuples(index=False):
        matrix.loc[row.gate, row.strategy] = mapping[row.gate_result]
        label_matrix.loc[row.gate, row.strategy] = row.gate_result
    display = matrix.fillna(-1).to_numpy()
    cmap = ListedColormap(["#F2F2F2", "#C95D63", "#D8B24C", "#4A9B73"])
    encoded = display + 1

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.imshow(encoded, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks([0, 1], ["Hybrid pan-ErbB\nvertical blockade", "Mesenchymal vascular\nremodeling + IO"])
    ax.set_yticks(np.arange(len(gate_order)), [g.replace("_", " ") for g in gate_order])
    for i in range(len(gate_order)):
        for j in range(2):
            label = label_matrix.iloc[i, j] if label_matrix.iloc[i, j] else "not tested"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.4,
                    color="white" if label in {"PASS", "FAIL"} else "#333333", fontweight="bold")
    ax.set_title("Multiscale qualification gates", fontweight="bold", pad=12)
    ax.set_xlabel("Therapeutic strategy")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.text(0.5, 0.01, "Gray: not evaluated in that strategy  |  Green: pass  |  Red: fail  |  Amber: unresolved",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(fig, FIG_MAIN / "Figure_2_multiscale_therapy_qualification")


def figure_3_stability_and_immune(stability: pd.DataFrame, enrichment: pd.DataFrame) -> list[str]:
    cohort_order = ["TCGA_BRCA", "METABRIC", "SCANB_GSE96058"]
    formula_order = list(FORMULA_LABELS)
    colors = ["#D49B3F", "#B87A2C", "#A94B5A"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    x = np.arange(len(cohort_order))
    width = 0.23
    for i, (formula, color) in enumerate(zip(formula_order, colors)):
        values = stability.set_index(["cohort", "formula"])["min_spearman"]
        y = [values.loc[(cohort, formula)] for cohort in cohort_order]
        ax.bar(x + (i - 1) * width, y, width=width, color=color, label=FORMULA_LABELS[formula])
    ax.axhline(0.70, color="#444444", linestyle="--", linewidth=1, label="Prespecified gate")
    ax.set_xticks(x, ["TCGA-BRCA", "METABRIC", "SCAN-B"])
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("Minimum Spearman correlation")
    ax.set_title("Weight sensitivity across cohorts", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=1)
    panel_label(ax, "A")

    ax2 = axes[1]
    immune_order = ["immune_desert_cold", "intermediate", "immune_excluded", "inflamed_dysfunctional", "hot_effective"]
    avg = enrichment.groupby(["formula", "immune_state"], as_index=False)["enrichment_ratio"].mean()
    heat = avg.pivot(index="formula", columns="immune_state", values="enrichment_ratio").reindex(
        index=formula_order, columns=immune_order
    )
    heat_vmax = max(2.5, float(np.ceil(np.nanmax(heat.to_numpy()) * 10.0) / 10.0))
    im = ax2.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=0, vmax=heat_vmax, aspect="auto")
    ax2.set_xticks(np.arange(len(immune_order)), [x.replace("_", "\n") for x in immune_order], rotation=25, ha="right")
    ax2.set_yticks(np.arange(len(formula_order)), [FORMULA_LABELS[x] for x in formula_order])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            ax2.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if value > 0.72 * heat_vmax else "black")
    ax2.set_title("Top-decile immune-state enrichment", loc="left", fontweight="bold")
    ax2.tick_params(length=0)
    ax2.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.72, pad=0.02)
    cbar.set_label("Mean enrichment ratio")
    panel_label(ax2, "B")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(fig, FIG_MAIN / "Figure_3_hypothesis_stability_and_immune_context")


def support_score(text: str) -> tuple[int, str]:
    value = str(text).lower()
    if value in {"supported", "global-fdr erbb anchor with class replication"}:
        return 3, "strong"
    if any(token in value for token in ["direct breast preclinical", "validated erbb anchor", "mechanism-consistent"]):
        return 3, "strong"
    if any(token in value for token in ["partly supported", "mechanistically coherent", "class precedent", "directionally robust"]):
        return 2, "partial"
    if any(token in value for token in ["nominal", "component-level", "not evaluable", "less state-selective", "insufficient"]):
        return 1, "limited"
    if any(token in value for token in ["none", "no robust", "not established", "unvalidated"]):
        return 0, "absent"
    return 1, "limited"


def figure_4_evidence_atlas(atlas: pd.DataFrame) -> list[str]:
    strategies = atlas[atlas["strategy_id"].ne("epithelial_unresolved")].copy()
    columns = [
        ("bulk_pathway_support", "Bulk\npathways"),
        ("malignant_single_cell_support", "Malignant\nsingle-cell"),
        ("spatial_ecosystem_support", "Spatial\necosystem"),
        ("target_dependency_support", "Target\ndependency"),
        ("perturbation_support", "Perturbation"),
        ("combination_dataset_support", "Combination\ndataset"),
        ("exact_pair_external_anchor", "External\npair anchor"),
    ]
    matrix, labels = [], []
    for row in strategies.itertuples(index=False):
        row_scores, row_labels = [], []
        for column, _ in columns:
            score, label = support_score(getattr(row, column))
            row_scores.append(score)
            row_labels.append(label)
        matrix.append(row_scores)
        labels.append(row_labels)
    matrix = np.asarray(matrix)
    cmap = ListedColormap(["#D9D9D9", "#E8C7A6", "#E3B64B", "#4A9B73"])

    fig, ax = plt.subplots(figsize=(10.7, 4.8))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), [label for _, label in columns])
    strategy_labels = [
        "Hybrid: pan-ErbB + AKT  [B1]",
        "Hybrid: pan-ErbB + mTORC1/2  [B2]",
        "Mesenchymal: vascular remodeling + IO  [B2E]",
        "Hybrid: pan-ErbB + IO  [C]",
    ]
    ax.set_yticks(np.arange(len(strategies)), strategy_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.2,
                    color="white" if matrix[i, j] == 3 else "#222222", fontweight="bold")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Final state-specific therapeutic evidence atlas", fontweight="bold", pad=12)
    fig.text(0.5, 0.02,
             "Evidence grades prioritize experimental testing; they do not denote clinical efficacy or predictive validity.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save_figure(fig, FIG_MAIN / "Figure_4_final_state_specific_evidence_atlas")


def figure_5_prospective_design() -> list[str]:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    steps = [
        (0.3, 3.15, 2.0, 1.2, "Models", "4 core Hybrid\n2 state-anchor controls", "#E6A23C"),
        (2.7, 3.15, 2.0, 1.2, "Pairs", "sapitinib + capivasertib\nneratinib + capivasertib", "#5B7FA3"),
        (5.1, 3.15, 2.0, 1.2, "Replication", "3 biological repeats\n2 technical matrices/plate", "#6F8F72"),
        (7.5, 3.15, 2.0, 1.2, "Primary screen", "36 locked 96-well plates\n5×5 non-zero grid", "#8A6FA8"),
        (9.9, 3.15, 1.8, 1.2, "Primary metric", "Mean Bliss excess\nacross full grid", "#A94B5A"),
    ]
    for x, y, w, h, title, body, color in steps:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                               facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", color=color, fontweight="bold", fontsize=10)
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=8.2)
    for x in [2.35, 4.75, 7.15, 9.55]:
        ax.annotate("", xy=(x + 0.28, 3.75), xytext=(x, 3.75), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    gate_text = (
        "Plate QC: vehicle CV ≤20% and Z′ ≥0.5   •   Model-positive: median Bliss >10, bootstrap 95% CI >0, "
        "positive direction in all repeats   •   Pair advance: ≥3/4 core models, including HER2+ and TNBC"
    )
    gate = FancyBboxPatch((0.5, 1.25), 11.0, 1.0, boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor="#F3F5F7", edgecolor="#777777", linewidth=1.2)
    ax.add_patch(gate)
    ax.text(6.0, 1.75, textwrap.fill(gate_text, 150), ha="center", va="center", fontsize=8.5)
    ax.text(6.0, 0.55,
            "Prospective and outcome-blind: templates, randomization seed, analysis rules, and mechanism gates were frozen before response data.",
            ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Locked prospective validation design for the primary Hybrid hypothesis", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return save_figure(fig, FIG_MAIN / "Figure_5_locked_prospective_validation_design")


def supplementary_negative_figure(negative: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    y = np.arange(len(negative))
    ax.scatter(np.zeros(len(y)), y, s=85, color="#A94B5A", zorder=3)
    ax.axvline(0, color="#A94B5A", linewidth=1)
    ax.set_yticks(y, [textwrap.fill(x, 38) for x in negative["negative_or_null_result"]])
    ax.set_xlim(-0.1, 1.0)
    ax.set_xticks([0], ["0 passed primary gate"])
    ax.invert_yaxis()
    ax.set_title("Important negative evidence retained in the final model", fontweight="bold")
    for i, interpretation in enumerate(negative["interpretation"]):
        ax.text(0.06, i, textwrap.fill(interpretation, 65), va="center", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return save_figure(fig, FIG_SUPP / "Figure_S1_important_negative_evidence")


def copy_tables() -> list[str]:
    mapping = {
        ATLAS_DIR / "final_state_specific_therapy_atlas.tsv": TAB_MAIN / "Table_1_final_state_specific_therapy_atlas.tsv",
        ATLAS_DIR / "final_state_biology_summary.tsv": TAB_MAIN / "Table_2_final_state_biology_summary.tsv",
        ATLAS_DIR / "final_claim_boundary_matrix.tsv": TAB_SUPP / "Table_S1_claim_boundary_matrix.tsv",
        ATLAS_DIR / "evidence_grade_legend.tsv": TAB_SUPP / "Table_S2_evidence_grade_legend.tsv",
        ATLAS_DIR / "important_negative_evidence.tsv": TAB_SUPP / "Table_S3_important_negative_evidence.tsv",
        ANALYSIS / "26_multiscale_state_therapy_qualification/multiscale_gate_evidence.tsv": TAB_SUPP / "Table_S4_multiscale_gate_evidence.tsv",
        ANALYSIS / "21_state_specific_therapy_scores/therapy_score_stability_summary.tsv": TAB_SUPP / "Table_S5_therapy_score_stability.tsv",
        ANALYSIS / "27_hybrid_compound_model_lock/locked_hybrid_compound_pairs.tsv": TAB_SUPP / "Table_S6_locked_hybrid_compound_pairs.tsv",
        ANALYSIS / "28_hybrid_plate_analysis_lock/combination_analysis_lock.tsv": TAB_SUPP / "Table_S7_prospective_combination_analysis_lock.tsv",
    }
    outputs = []
    for source, destination in mapping.items():
        shutil.copy2(source, destination)
        outputs.append(str(destination))
    return outputs


def write_manuscript_modules(
    pathway_summary: dict,
    pharmacogenomics_summary: dict,
    dependency_summary: dict,
    drugcomb_summary: dict,
    robustness_summary: dict,
    multiscale_summary: dict,
    plate_summary: dict,
) -> list[str]:
    results = f"""# Results module: EMT-state-specific therapeutic evidence synthesis

## EMT states resolve into distinct biological architectures

Redundancy-collapsed pathway analysis identified {pathway_summary['representatives_after_collapse']} multiscale-supported pathway representatives: {pathway_summary['representatives_by_state']['epithelial-enriched']} Epithelial-enriched, {pathway_summary['representatives_by_state']['hybrid-peak']} Hybrid-peak, and {pathway_summary['representatives_by_state']['mesenchymal-enriched']} Mesenchymal-enriched programs. Epithelial tumors were dominated by metabolic adaptation and protein/RNA homeostasis, Hybrid tumors concentrated adhesion/motility and growth-survival signaling, and Mesenchymal tumors combined ECM remodeling, inflammatory signaling, and hypoxia/angiogenesis. These differences argued against a universal EMT-directed regimen (Figure 1; Table 2).

## ERBB signaling provides the strongest malignant-cell anchor in the Hybrid state

The target-first dependency screen evaluated {dependency_summary['unique_drug_state_tests']} drug-state tests and retained a globally corrected Hybrid ERBB dependency anchor. In contrast, the locked candidate pharmacogenomic analysis produced {pharmacogenomics_summary['global_fdr_lt_0.10']} results at global FDR <0.10, emphasizing that isolated sensitivity signals were insufficient for therapy promotion. Integration of malignant single-cell identity, within-patient programs, and paired spatial evidence qualified Hybrid pan-ErbB vertical blockade across all nine prespecified multiscale gates (Figure 2).

## Combination datasets prioritize hypotheses but do not establish efficacy

DrugComb analysis tested {drugcomb_summary['locked_pairs_tested']} locked pairs; none passed the primary global FDR threshold, although two pairs were nominally significant. Secondary Hybrid robustness analysis retained {robustness_summary['directionally_robust_pairs']} directionally robust pairs, but neither subtype-adjusted nor class-level tests passed FDR <0.10. Accordingly, pan-ErbB plus AKT inhibition was classified as the primary biomarker-defined experimental hypothesis, whereas pan-ErbB plus dual mTORC1/2 inhibition remained a secondary, anchor-specific hypothesis (Figure 4; Table 1).

## Mesenchymal therapy requires an ecosystem rather than tumor-cell-only model

Mesenchymal tumors showed malignant-state identity, T-cell proximity, antigen visibility, checkpoint engagement, stromal adjacency, and incomplete cytotoxic execution. However, the functional stromal-barrier gate failed and vascular normalization was not measured. The resulting antiangiogenic-plus-checkpoint strategy is therefore a mechanistic ecosystem-remodeling hypothesis that requires a vascularized immune model, not a validated drug pair (Figure 2; Table 1).

## Hypothesis scores are stable ranking tools, not response predictors

Outcome-blind therapy-hypothesis scores were calculated in 6,332 tumors from TCGA-BRCA, METABRIC, and SCAN-B. All prespecified weight-sensitivity gates passed, with minimum within-cohort Spearman correlations above 0.80 for every strategy. Mesenchymal vascular-remodeling scores preferentially enriched inflamed-dysfunctional and immune-excluded contexts, whereas Hybrid scores captured a broader mixture of immune states (Figure 3). Because no treatment outcome was used and weights were not fitted, these scores rank experimental eligibility but do not predict treatment response.

## Prospective validation was locked before response data

The primary Hybrid program comprises four core models, two state-anchor controls, two pan-ErbB/AKT pairs, and three biological repeats, yielding {plate_summary['primary_screen_plates']} locked 96-well plates. The primary estimand is mean Bliss excess across the complete 5×5 non-zero dose grid; plate QC, bootstrap confidence intervals, model-positive criteria, pair-advance rules, and mechanistic phosphoprotein gates were frozen before response data (Figure 5). No treatment-predictive biomarker, combination efficacy, or clinical recommendation is claimed.
"""
    results_path = MANUSCRIPT / "RESULTS_STATE_SPECIFIC_THERAPY_ATLAS.md"
    results_path.write_text(results, encoding="utf-8")

    legends = """# Figure legends

## Figure 1. EMT states define distinct biological and therapeutic contexts

(A) Nonredundant multiscale pathway representatives grouped by biological theme for Epithelial-enriched, Hybrid-peak, and Mesenchymal-enriched tumors. Unresolved “other” pathways are omitted from the visualization but retained in the source table. (B) State-specific therapeutic positioning. Hybrid tumors are prioritized for ERBB vertical-blockade experiments, Mesenchymal tumors require vascularized immune modeling, and no Epithelial combination is promoted.

## Figure 2. Multiscale qualification of prioritized strategies

Qualification gates integrating malignant single-cell programs, within-patient coupling, paired spatial evidence, and dependency anchors. Gray cells were not evaluated for the corresponding strategy. The Mesenchymal vascular-normalization mechanism remains unresolved, and the functional stromal-barrier gate did not pass.

## Figure 3. Stability and immune context of therapy-hypothesis scores

(A) Minimum Spearman correlation across the prespecified weight grid in TCGA-BRCA, METABRIC, and SCAN-B. The dashed line marks the 0.70 stability gate. (B) Mean enrichment ratio of immune states among the top decile of each therapy-hypothesis score across cohorts. Scores were derived without treatment outcomes and are prioritization tools rather than response predictors.

## Figure 4. Final state-specific therapeutic evidence atlas

Evidence-layer synthesis for four promoted or exploratory strategies. Strong, partial, limited, and absent evidence are displayed across pathway, single-cell, spatial, dependency, perturbational, combination-dataset, and external-pair layers. Grades reflect experimental readiness and do not indicate clinical efficacy.

## Figure 5. Locked prospective validation design for the primary Hybrid hypothesis

The prospective program includes four core Hybrid models, two state-anchor controls, two pan-ErbB/AKT pairs, three biological repeats, two technical matrices per plate, and 36 locked 96-well plates. Plate QC and pair-advance rules were frozen before response data.

## Figure S1. Important negative evidence retained in the final model

Primary corrected gates not passed by global state-selective pharmacogenomic candidates, DrugComb pairs, Hybrid robustness tests, the compact malignant-cell selector, or an Epithelial combination. Negative results constrain the final claim set and prevent universal or efficacy-level conclusions.
"""
    legends_path = MANUSCRIPT / "FIGURE_LEGENDS_STATE_THERAPY.md"
    legends_path.write_text(legends, encoding="utf-8")
    return [str(results_path), str(legends_path)]


def main() -> None:
    for directory in [FIG_MAIN, FIG_SUPP, TAB_MAIN, TAB_SUPP, REPORT_DIR, MANUSCRIPT]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()

    themes = pd.read_csv(ANALYSIS / "06_collapsed_state_pathway_map/state_biological_theme_counts.tsv", sep="\t")
    states = pd.read_csv(ATLAS_DIR / "final_state_biology_summary.tsv", sep="\t")
    gates = pd.read_csv(ANALYSIS / "26_multiscale_state_therapy_qualification/multiscale_gate_evidence.tsv", sep="\t")
    stability = pd.read_csv(ANALYSIS / "21_state_specific_therapy_scores/therapy_score_stability_summary.tsv", sep="\t")
    enrichment = pd.read_csv(ANALYSIS / "21_state_specific_therapy_scores/top_decile_immune_state_enrichment.tsv", sep="\t")
    atlas = pd.read_csv(ATLAS_DIR / "final_state_specific_therapy_atlas.tsv", sep="\t")
    negative = pd.read_csv(ATLAS_DIR / "important_negative_evidence.tsv", sep="\t")

    outputs = []
    outputs += figure_1_state_biology(themes, states)
    outputs += figure_2_multiscale_gates(gates)
    outputs += figure_3_stability_and_immune(stability, enrichment)
    outputs += figure_4_evidence_atlas(atlas)
    outputs += figure_5_prospective_design()
    outputs += supplementary_negative_figure(negative)
    outputs += copy_tables()

    pathway_summary = json.loads((ANALYSIS / "06_collapsed_state_pathway_map/collapsed_state_pathway_map_summary.json").read_text(encoding="utf-8"))
    pharmacogenomics_summary = json.loads((ANALYSIS / "11_pharmacogenomic_validation/pharmacogenomic_validation_summary.json").read_text(encoding="utf-8"))
    dependency_summary = json.loads((ANALYSIS / "13_target_first_dependency_screen/target_first_dependency_screen_summary.json").read_text(encoding="utf-8"))
    drugcomb_summary = json.loads((ANALYSIS / "17_state_specific_drugcomb_synergy/state_specific_drugcomb_synergy_summary.json").read_text(encoding="utf-8"))
    robustness_summary = json.loads((ANALYSIS / "18_hybrid_drugcomb_robustness/hybrid_drugcomb_robustness_summary.json").read_text(encoding="utf-8"))
    multiscale_summary = json.loads((ANALYSIS / "26_multiscale_state_therapy_qualification/multiscale_state_therapy_qualification_summary.json").read_text(encoding="utf-8"))
    plate_summary = json.loads((ANALYSIS / "28_hybrid_plate_analysis_lock/hybrid_plate_analysis_lock_summary.json").read_text(encoding="utf-8"))
    outputs += write_manuscript_modules(
        pathway_summary, pharmacogenomics_summary, dependency_summary, drugcomb_summary,
        robustness_summary, multiscale_summary, plate_summary,
    )

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "paper-ready state-therapy figure, table, and Results package",
        "main_figures": 5,
        "supplementary_figures": 1,
        "main_tables": 2,
        "supplementary_tables": 7,
        "figure_formats": ["PNG_600dpi", "PDF_vector", "SVG_vector"],
        "manuscript_modules": [
            "RESULTS_STATE_SPECIFIC_THERAPY_ATLAS.md",
            "FIGURE_LEGENDS_STATE_THERAPY.md",
        ],
        "validated_treatment_predictive_biomarker": False,
        "validated_combination_efficacy": False,
        "generated_files": outputs,
        "next_gate": "visual QC of all figures, then integrate with the earlier EMT-immune biological landscape into the full manuscript architecture",
    }
    summary_path = REPORT_DIR / "final_paper_package_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "generated_files"}, indent=2))


if __name__ == "__main__":
    main()
