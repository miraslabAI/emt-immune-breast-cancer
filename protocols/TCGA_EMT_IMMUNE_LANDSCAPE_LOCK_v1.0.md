# TCGA EMT–Immune Landscape Lock v1.0

- Lock date: 2026-10-31
- Analysis: outcome-blind biological discovery
- Cohort: TCGA-BRCA, one primary tumour per patient
- Survival, recurrence and treatment outcomes: prohibited

## Primary EMT axes

1. epithelial identity: `candidate_EMP::Tan2014_E`;
2. mesenchymal core: standardized mean of `Tan2014_M`, `Hallmark_EMT`,
   `EMT-II` and `EMT-III`;
3. EMT position: mesenchymal core minus epithelial identity;
4. E/M coactivation: minimum of the within-cohort percentile ranks of epithelial
   identity and mesenchymal core.

Coactivation is a bulk hybrid/plasticity proxy, not temporal dynamics.

## Primary immune axes

1. antigen presentation: Steps 1 and 2-positive;
2. priming/recruitment: Step 3-positive plus T-cell, CD8, Th1, dendritic and NK
   recruitment;
3. infiltration: Step 5-positive minus Step 5-negative;
4. execution: mean Steps 6/7-positive minus mean Steps 6/7-negative;
5. suppressive recruitment: Treg, MDSC, macrophage and neutrophil recruitment;
6. overall activation: mean of antigen presentation, priming/recruitment,
   infiltration and execution;
7. immune exclusion pressure: mean Step 5-negative and suppressive recruitment.

## Analyses

- Primary: Spearman correlations between four EMT and seven immune axes.
- Sensitivity: partial rank correlations adjusted for PAM50 and xCell stromal
  composite where available.
- Secondary: complete retained EMT-by-TIP correlation matrix.
- Effect modification: correlations within PAM50 subtype; no favorable-subgroup
  selection.
- Nonlinearity: quadratic EMT-position and coactivation terms for each immune
  axis.
- Uncertainty: 1,000 patient-bootstrap replicates for primary correlations.
- Multiplicity: Benjamini-Hochberg FDR within each prespecified analysis family.

## Descriptive immune states

Continuous axes remain primary. Outcome-blind median-zero standardized rules
create mutually exclusive visualization labels: immune desert/cold,
immune-excluded, inflamed-effective, inflamed-dysfunctional and intermediate.
These labels are not used as primary predictors or optimized against outcome.

Stromal adjustment is a sensitivity analysis rather than proof of tumour purity.
A later single-cell/spatial phase must establish cell of origin.

