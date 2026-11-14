# Multiscale EMT–immune synthesis lock v1.0

## Scope

This synthesis is outcome-free. It integrates the already completed bulk, single-cell malignant-cell, and baseline spatial analyses. NeoTRIP `pCR` and `Arm` must not be read here.

## Biological question

Do epithelial, mesenchymal-dominant, hybrid E/M, and double-low malignant states occupy reproducibly different immune ecosystems, and which biological bottleneck—not merely which prognostic association—is supported for each state?

## Evidence hierarchy

1. Three-cohort bulk replication (TCGA-BRCA, METABRIC, SCAN-B): population-level reproducibility.
2. GSE176078 malignant-cell analysis: tumor-cell-intrinsic programs and within-patient state variation.
3. NeoTRIP baseline spatial proteomics: local cellular architecture and functional phenotype of neighboring cells.

No single layer is sufficient for a therapy claim. A mechanism is considered convergent only when at least two independent layers support compatible biology. Absence of significance is reported as unresolved, not evidence of absence.

## Locked state interpretations to evaluate

- **Epithelial-dominant:** primarily an immune-initiation/visibility-poor or cold state.
- **Mesenchymal-dominant:** immune-engaged but functionally ineffective, with myeloid/stromal proximity and impaired local T-cell execution.
- **Hybrid E/M:** immune-visible/recruiting and simultaneously barrier-forming; a distinct state, not an intermediate average.
- **Double-low:** biologically heterogeneous/indeterminate; purity or dilution remains a competing explanation.

## Multiplicity and reporting

- Existing layer-specific FDR values are retained; no cross-layer re-testing is performed.
- Spatial effects use paired patient contrasts against epithelial-dominant cells.
- Exact effect, 95% bootstrap CI, FDR, and paired-patient count are retained.
- Therapy implications are labeled hypothesis-generating and must be tested against treatment interaction—not inferred from pCR association alone.

## Gate before treatment outcomes

Treatment outcomes may be opened only after the synthesis outputs pass and a separate treatment-context lock specifies: endpoint, arm coding, a small fixed set of mechanism scores, interaction estimands, covariates, missing-data policy, multiplicity family, and sensitivity analyses.

