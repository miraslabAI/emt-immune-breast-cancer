# Bulk EMT--immune landscape replication lock v1.0

The TCGA axis definitions in `TCGA_EMT_IMMUNE_LANDSCAPE_LOCK_v1.0.md` are
carried forward unchanged to METABRIC and SCAN-B/GSE96058. No clinical outcome
is read. Scores are standardized within cohort because the platforms differ.

## Primary replication

- The same 4 EMT axes by 7 immune axes (28 associations) are evaluated with
  Spearman correlation and 1,000 patient bootstrap intervals in each cohort.
- The TCGA, METABRIC, and SCAN-B estimates are combined by a fixed-effect
  Fisher-z meta-analysis weighted by n-3.
- Replication strength is described by cross-cohort sign concordance, whether
  all three bootstrap intervals exclude zero in the same direction, the pooled
  confidence interval, and heterogeneity (Q and I-squared).
- FDR is controlled within each cohort across the 28 locked primary tests.

## Sensitivity and phenotypes

- Partial rank correlations adjust for PAM50 and the xCell stromal composite
  among complete cases.
- The locked descriptive cold, immune-excluded, hot-effective,
  inflamed-dysfunctional, and intermediate rules are applied after within-cohort
  standardization.
- Phenotype frequencies are descriptive and must not be interpreted as response
  rates or outcome classes.

No axes, thresholds, signs, or subsets may be changed after results are opened.

