# NeoTRIP outcome-free spatial mechanism lock v1.0

## Objective

Determine whether epithelial, mesenchymal, hybrid E/M and double-low tumour-cell
states occupy distinct immune neighbourhoods at baseline. No treatment arm or
pCR value is read or evaluated.

## EMT states

Invasive epithelial cells are scored with the previously frozen patient-balanced
NeoTRIP calibration for Vimentin, panCK, CK8/18 and CK5/14. Epithelial strength
is the mean of the three keratin z-scores. States use zero on the frozen scales:

- epithelial-dominant: epithelial strength >= 0 and Vimentin < 0;
- mesenchymal-dominant: Vimentin >= 0 and epithelial strength < 0;
- hybrid E/M: both >= 0;
- double-low: both < 0.

## Spatial neighbourhoods

Within each FOV, each invasive epithelial cell is assigned its 20 nearest cells,
excluding itself. No edge may cross an FOV. T, NK, Mono/Mac and Stroma neighbour
fractions are normalized to their availability in the same FOV and expressed as
log2 enrichment with a fixed 0.01 pseudocount.

Additional neighbour states are T cytotoxicity (GZMB/T-bet), T dysfunction
(PD-1/TOX), Mono/Mac suppression (CD163, two PD-L1 antibodies and IDO), and
stromal barrier activity (SMA, PDGFRB, PDPN and Calponin). Markers are z-scored
within patient and relevant lineage. Tumour-cell HLA-ABC and PD-L1 are secondary
target-cell measurements.

## Inference

FOV/state summaries are aggregated to patient/state summaries, weighted by the
number of target tumour cells. Each non-epithelial state is paired against the
epithelial-dominant state within patient. Effects use 10,000 patient-level sign
permutations and 2,000 patient bootstraps. At least 10 paired patients are required;
FDR is controlled across the locked contrasts. Cell counts are never used as the
independent inferential sample size.

This is mechanistic spatial evidence, not proof of treatment efficacy.

