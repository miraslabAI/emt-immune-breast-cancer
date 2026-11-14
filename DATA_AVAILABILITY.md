# Data availability and expected placement

Raw/patient-level data are not included in this repository.

| Resource | Stable identifier/source | Manuscript role |
|---|---|---|
| TCGA-BRCA | GDC project `TCGA-BRCA` | bulk discovery |
| METABRIC | cBioPortal study `brca_metabric` | bulk replication |
| SCAN-B | GEO `GSE96058` | bulk replication |
| Breast cancer single-cell atlas | GEO `GSE176078` | malignant-cell qualification |
| NeoTRIP baseline spatial data | Zenodo `10.5281/zenodo.14290163`, `nt_preprocessed.h5ad`, version 1 | spatial qualification |
| MSigDB Hallmark | `h.all.v2026.1.Hs.symbols.gmt` | pathway analysis |
| Reactome | human pathway release accessed 2026-10-07 | pathway analysis |
| LINCS L1000 Level 5 | GEO `GSE70138` / Connectivity Map | perturbational prioritization |
| PRISM | DepMap PRISM Repurposing 20Q2 | pharmacogenomic support |
| GDSC | GDSC dose-response/model annotations | pharmacogenomic support |
| DrugComb | version 1.5 | combination evidence audit |

The user is responsible for observing each resource's access terms. No outcome data were used to define EMT states, signatures, scores, or drug ranks in the reported outcome-blind workflow.
