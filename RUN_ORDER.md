# Manuscript-aligned run order

The scripts preserve the paper's inferential sequence: population replication first, then cellular and spatial qualification, then pathway and therapeutic prioritization.

1. **Data audit and preprocessing** — `code/02_quality_control`, `code/03_preprocessing`, and `code/04_feature_engineering`.
2. **Bulk discovery** — `build_tcga_emt_immune_landscape.py`.
3. **Locked bulk replication** — `replicate_emt_immune_landscape_bulk.py`.
4. **State/immune archetypes** — `build_emt_state_immune_archetypes.py`.
5. **Malignant-cell origin** — `analyze_gse176078_cell_origin.py`.
6. **Spatial mechanisms and topology** — `analyze_neotrip_spatial_mechanisms.py`, then `analyze_neotrip_spatial_emt_topology.py`.
7. **Multiscale synthesis** — `build_multiscale_emt_immune_synthesis.py` and `build_final_mechanistic_therapy_matrix.py`.
8. **State pathways** — run the pathway, replication, redundancy-collapse and target-mapping scripts in `code/10_drug_repositioning` in filename/logical dependency order.
9. **Drug evidence gates** — run LINCS, dependency, pharmacogenomic and DrugComb qualification scripts. Retain null results.
10. **Final atlas** — `build_final_state_specific_therapy_atlas.py`.
11. **Paper package** — `code/13_figures_tables/build_final_paper_package.py`.

Scripts that ingest future wet-laboratory results are included only to preserve the prospective contract and should not be run as part of reproducing the current paper.
