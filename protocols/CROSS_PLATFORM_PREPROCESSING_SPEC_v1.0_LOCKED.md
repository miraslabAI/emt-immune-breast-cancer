# Cross-platform preprocessing specification v1.0 — LOCKED

## Status and scope

- Lock date: 2026-10-31
- Scope: primary bulk survival and neoadjuvant pCR branches
- Outcome use during specification: none
- Superseded only by a versioned amendment that records the reason and timing
- Cross-cohort batch correction is prohibited.

This document fixes the transformations that occur before feature construction or model fitting. Cohorts remain independent; no ComBat, pooled quantile normalization, outcome-guided filtering, or validation-derived scaling is allowed.

## Locked cohort roles

| Branch | Cohort | Role | Locked analysis count |
|---|---|---|---:|
| Survival | METABRIC | Development | 1,979 |
| Survival | TCGA-BRCA | External validation 1 | 1,080 OS; 830 PAM50-complete |
| Survival | SCAN-B/GSE96058 | External validation 2 | 3,273 |
| pCR | GSE25055 | Discovery | 306 |
| pCR | GSE25065 | Locked validation | 182 |

## Gene identifiers and duplicate handling

1. All features use uppercase HGNC-style gene symbols.
2. TCGA Ensembl versions are removed and mapped with GENCODE v50.
3. Blank, unmapped, or ambiguous identifiers are discarded and logged.
4. Duplicate gene rows in gene-level bulk matrices are collapsed by within-sample median. This rule is outcome-blind.
5. SCAN-B columns F1–F3273 are linked to GSM accessions using the frozen metadata mapping. The 136 technical/extra replicate columns are excluded from primary analyses.
6. GSE25055/GSE25065 use the locked GPL96 mapping. Only probes annotated uniquely to one gene are eligible.
7. When multiple GPL96 probes map to one gene, the probe with the largest median absolute deviation in GSE25055 is used. Ties use lexical probe ID. GSE25065 expression and outcomes do not influence selection.
8. The locked GPL96 map contains 12,501 genes and every selected probe is present in GSE25065.

## Measured gene universe

- Five-cohort common measured universe: 12,024 genes.
- Lock SHA-256: `d26aed49b2d7180c584fa9d3b2789e477a787aaa6ad9068bcf9e6c7f0696f9d8`
- This universe defines portability, not automatic inclusion in a predictive model.
- A signature requires at least 80% gene coverage in every cohort used for that confirmatory comparison.

The following TIP components fail the 80% rule on GPL96 and are excluded from the confirmatory pCR model: Step3-negative, CD4-T-cell recruiting, eosinophil recruiting, basophil recruiting, Step6-negative, and Step7-negative. They may be reported only as non-confirmatory measurement failures.

## Expression scale and sample-wise representation

1. Author-provided normalized/log-scale matrices are retained; no second quantile normalization is applied.
2. Values are checked for finite numbers, constant rows, duplicated samples, and implausible ranges before scoring.
3. For cross-platform signatures, genes are ranked within each sample over the eligible measured universe and converted to percentile ranks.
4. An unsigned set score is the mean percentile rank of its observed genes minus 0.5.
5. A signed score is mean percentile rank of up genes minus mean percentile rank of down genes.
6. No missing signature gene is imputed. Coverage is recorded per cohort and the score is rejected below 80%.
7. Predictive-feature centering/scaling is fitted only in the development cohort and then frozen:
   - survival: fit in METABRIC and apply unchanged to TCGA and SCAN-B;
   - response: fit in GSE25055 and apply unchanged to GSE25065.
8. Zero-variance features in development are removed before resampling and recorded. Validation variance is never used for feature removal.

## Clinical harmonization

### Survival primary clinical core

- Age at diagnosis: continuous; modelled with a restricted cubic spline only if the knot locations are fixed within METABRIC before validation.
- Tumor burden: harmonized ordinal T category. Numeric tumor size uses T1 ≤20 mm, T2 >20–50 mm, T3 >50 mm; source T categories retain T1–T4, with T4 kept distinct when supported.
- Node status: N0 versus node-positive. NX/unknown is missing.

PAM50 is not imputed. Models containing PAM50 use the PAM50-complete analysis set. Canonical labels are LumA, LumB, Basal, HER2-enriched, Normal-like, and Other/Claudin-low. Grade is excluded from the primary survival model because TCGA lacks a portable field; it is a METABRIC/SCAN-B sensitivity covariate.

### pCR clinical comparator

- Age, clinical T, clinical N, grade, ER, PR, HER2, and PAM50.
- All passed 80% coverage in discovery and locked validation.
- Categories are harmonized using discovery-defined dictionaries before validation is opened.

## Missing-data policy

1. PAM50 is never imputed.
2. Continuous clinical variables use median imputation fitted within each discovery resample.
3. Categorical variables use the discovery-resample mode plus an explicit missing indicator when development missingness is at least 2%.
4. The final imputation constants are refitted on the full development cohort after tuning and applied unchanged to validation.
5. Complete-case results are mandatory sensitivity analyses.
6. Validation outcomes are not used to select an imputation rule, threshold, feature, or transformation.

## Leakage controls

- Every filter, imputer, scaler, feature constructor, and tuning operation is fitted inside development resampling.
- Patient-grouped resampling is mandatory.
- GSE25065 outcomes remain sealed until the final response pipeline is serialized and hashed.
- TCGA and SCAN-B do not participate in survival tuning.
- Any accidental outcome exposure requires an incident entry and downgrades the affected result to exploratory.

## Required preprocessing outputs

Each cohort must produce a matrix, sample manifest, feature dictionary, coverage table, transformation parameters, SHA-256 hashes, software versions, and a QC summary. A cohort cannot enter modelling until these artifacts agree on sample order and feature order.
