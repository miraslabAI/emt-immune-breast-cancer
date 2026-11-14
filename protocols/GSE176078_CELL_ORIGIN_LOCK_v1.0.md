# GSE176078 EMT--immune cell-origin analysis lock v1.0

## Purpose

Test whether the cross-cohort bulk EMT--immune archetypes have malignant-cell
and patient-level microenvironment support. This is an outcome-free mechanistic
analysis, not a treatment-efficacy analysis.

## Units of inference

- Primary unit: patient (20 patients with malignant cells).
- Secondary unit: malignant metacell, with patient-clustered standard errors and
  patient fixed effects. Metacells are never treated as independent patients.
- Cell-count-weighted summaries use the number of malignant cells represented by
  each metacell.

## Malignant EMT features

Epithelial and mesenchymal program medians are calculated within clinical subtype
to define epithelial-dominant, mesenchymal-dominant, hybrid E/M and double-low
metacells. Patient features are weighted mean EMT position, mesenchymal occupancy,
hybrid occupancy, within-tumour EMT-position dispersion and four-state entropy.

## Tumour-intrinsic immune programmes

Locked gene sets assess antigen presentation, IFN response, checkpoint ligands,
T/NK-recruiting chemokines, myeloid-recruiting factors and TGF-beta/ECM signalling.
Scores are means of gene-wise standardized log-normalized metacell expression.
Gene coverage is reported and no gene set is redefined after results are opened.

## Patient-level microenvironment

The malignant EMT features are tested against patient fractions of T cells,
myeloid cells, CAFs, endothelial cells, B cells and PVL cells, plus a T-to-myeloid
log-ratio. Spearman effects use 10,000 patient-level permutations and 2,000
patient bootstraps. FDR is controlled across the locked feature-composition grid.
Subtype-adjusted partial-rank effects are a sensitivity analysis.

## Secondary within-patient analysis

Each tumour-intrinsic programme is modelled as a function of within-patient EMT
position and E/M coactivation, adjusting cell-cycle and stress ranks and patient
fixed effects, with patient-clustered standard errors.

Mechanistic therapy hypotheses require convergence with bulk, spatial and
treatment-context evidence; this analysis alone cannot nominate an effective drug.

