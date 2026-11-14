# EMT-state immune-archetype analysis lock v1.0

## Objective

Define reproducible EMT-state/immune-context combinations that can generate
mechanistic hypotheses for future combination-therapy studies. This analysis is
outcome-free and does not claim treatment efficacy.

## EMT states

Within each cohort, epithelial and mesenchymal-core scores are split at their
cohort medians. Every tumour is assigned to exactly one state:

- epithelial-dominant: epithelial high, mesenchymal low;
- mesenchymal-dominant: epithelial low, mesenchymal high;
- hybrid E/M: both high;
- double-low/indeterminate: both low.

Continuous EMT position and E/M coactivation remain primary quantitative axes;
the four states are an interpretable complementary representation.

## Immune characterization

Each EMT state is profiled for the seven locked immune axes and for individual
TIP components representing antigen presentation, priming, lymphoid recruitment,
myeloid/Treg recruitment, infiltration and execution. Scores are standardized
within cohort before comparison.

For each state/feature/cohort, report n, mean, median and a 1,000-patient-bootstrap
95% CI for the mean. Continuous decile profiles are generated for EMT position
and E/M coactivation to avoid relying solely on dichotomization.

## Joint archetypes

The four EMT states are crossed with the five previously locked descriptive
immune states. Cohort-specific Fisher tests and a cohort-stratified common odds
ratio quantify enrichment of each immune state within each EMT state. FDR is
controlled across the 20 locked combinations.

Results generate mechanistic priorities only. Therapy hypotheses require
cell-of-origin validation, spatial evidence and treatment-context evidence.

