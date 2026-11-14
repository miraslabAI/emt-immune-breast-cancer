# NeoTRIP spatial EMT topology lock v1.0

## Question

Does the hybrid E/M state form a spatial bridge between epithelial-dominant and mesenchymal-dominant malignant cells, and is intratumour EMT-state heterogeneity coupled to immune engagement or barrier biology?

## Scope

- Baseline NeoTRIP spatial proteomics; the same frozen eligible patients and marker calibration as the prior outcome-free analysis.
- `pCR` and `Arm` are not read.
- This is a cross-sectional spatial topology analysis. Contact patterns may be consistent with a state continuum, but they do not prove temporal transition, lineage direction, or plasticity rate.

## Malignant-cell states and graph

- The four previously frozen states are reused without alteration: epithelial-dominant, mesenchymal-dominant, hybrid E/M, and double-low.
- Within each FOV, each invasive epithelial cell is connected to its 20 nearest invasive epithelial neighbours, or all other malignant cells if fewer than 20 exist.
- FOVs with fewer than 10 classified malignant cells are excluded from topology estimation.
- State-to-state contact enrichment is observed directed edge count divided by the composition-derived expected edge count, reported on log2 scale with a 0.5-edge pseudocount.
- FOV results are combined to patients by summing observed and expected edge counts. The patient is the inference unit.

## Locked primary topology estimands

1. Symmetric epithelial–hybrid contact enrichment.
2. Symmetric hybrid–mesenchymal contact enrichment.
3. Symmetric direct epithelial–mesenchymal contact enrichment.
4. **Hybrid bridge contrast:** mean of epithelial–hybrid and hybrid–mesenchymal log2 enrichments minus direct epithelial–mesenchymal log2 enrichment.
5. Excess same-state contact fraction above composition-derived expectation.

Each mean is tested against zero using 10,000 patient-level sign permutations with 2,000 patient bootstraps. The five topology tests form one BH-FDR family.

## Heterogeneity–immune coupling

- Patient EMT heterogeneity is normalized Shannon entropy across the four states.
- A secondary bridge-strength score is the locked hybrid bridge contrast.
- Each is correlated with the ten previously computed overall spatial immune/barrier metrics using Spearman correlation, 10,000 patient-label permutations, and 2,000 patient bootstraps.
- Entropy and bridge-strength correlations use separate ten-test FDR families.

## Interpretation

- Positive hybrid bridge contrast supports a spatially intermediate/connecting topology, not a temporal transition claim.
- Positive entropy–barrier or bridge–barrier coupling supports ecological coexistence of plasticity and immune constraint.
- Null results leave the dynamic interpretation unresolved and must not be rescued by outcome-guided thresholds.

