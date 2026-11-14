#!/usr/bin/env python
"""Freeze result-ingestion templates and the prospective synergy-analysis contract."""
from __future__ import annotations

import os

import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOCK = ROOT / "04_analysis/11_drug_repositioning/28_hybrid_plate_analysis_lock"
OUT = ROOT / "04_analysis/11_drug_repositioning/29_hybrid_result_ingestion_contract"
RANDOMIZATION_SEED = 20261101


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(LOCK / "primary_plate_manifest.tsv", sep="\t")
    layout = pd.read_csv(LOCK / "locked_96well_layout.tsv", sep="\t")
    pairs = pd.read_csv(LOCK / "primary_pair_lock.tsv", sep="\t")
    models = pd.read_csv(LOCK / "primary_backup_model_lock.tsv", sep="\t")

    expected_plates = 36
    if len(manifest) != expected_plates or manifest["plate_id"].duplicated().any():
        raise ValueError("The stage-28 plate manifest is not the locked 36-plate design")
    if len(layout) != 96 or layout["well"].duplicated().any():
        raise ValueError("The stage-28 layout is not a unique 96-well design")

    rng = np.random.default_rng(RANDOMIZATION_SEED)
    order = pd.DataFrame({
        "plate_id": rng.permutation(manifest["plate_id"].to_numpy()),
        "randomized_processing_order": np.arange(1, len(manifest) + 1),
    })
    randomized_manifest = manifest.drop(columns=["randomized_processing_order"]).merge(
        order, on="plate_id", how="left", validate="one_to_one"
    ).sort_values("randomized_processing_order")
    randomized_manifest.to_csv(OUT / "randomized_primary_plate_manifest.tsv", sep="\t", index=False)

    pair_agents = pairs.set_index("pair_id")[["agent_A", "agent_B"]]
    raw_rows: list[dict] = []
    for plate in randomized_manifest.itertuples(index=False):
        agent_a, agent_b = pair_agents.loc[plate.pair_id].tolist()
        for well in layout.itertuples(index=False):
            raw_rows.append({
                "plate_id": plate.plate_id,
                "randomized_processing_order": plate.randomized_processing_order,
                "biological_repeat": plate.biological_repeat,
                "cell_line": plate.cell_line,
                "pair_id": plate.pair_id,
                "agent_A": agent_a,
                "agent_B": agent_b,
                "well": well.well,
                "technical_replicate": well.technical_replicate,
                "well_role": well.well_role,
                "agent_A_IC50_multiple": well.agent_A_IC50_multiple,
                "agent_B_IC50_multiple": well.agent_B_IC50_multiple,
                "raw_signal": np.nan,
                "read_status": "pending",
                "exclusion_reason": "",
                "operator_note": "",
            })
    raw = pd.DataFrame(raw_rows)
    if len(raw) != expected_plates * 96 or raw.duplicated(["plate_id", "well"]).any():
        raise ValueError("Raw-result template does not contain exactly 96 unique wells per plate")
    raw.to_csv(OUT / "raw_plate_results_template.tsv", sep="\t", index=False)

    metadata = randomized_manifest.copy()
    for column in [
        "run_date", "operator_id", "instrument_id", "assay_platform",
        "cell_seeding_density_per_well", "assay_duration_hours", "vehicle",
        "agent_A_stock_lot", "agent_B_stock_lot", "plate_note",
    ]:
        metadata[column] = ""
    metadata.to_csv(OUT / "plate_metadata_template.tsv", sep="\t", index=False)

    screening = models.loc[
        models["role"].isin(["core_primary", "state_anchor_control"]),
        ["cell_line", "role"],
    ].drop_duplicates()
    marker_plan = {
        "pERBB_pan": 6,
        "pPRAS40": 6,
        "pS6": 6,
        "cleaved_caspase3": 24,
        "cleaved_PARP": 24,
    }
    mechanism_rows = []
    for model, pair_id, repeat, condition, marker in itertools.product(
        screening.itertuples(index=False),
        pairs["pair_id"],
        range(1, 4),
        ["vehicle", "agent_A", "agent_B", "combination"],
        marker_plan,
    ):
        mechanism_rows.append({
            "cell_line": model.cell_line,
            "model_role": model.role,
            "pair_id": pair_id,
            "biological_repeat": repeat,
            "condition": condition,
            "marker": marker,
            "timepoint_hours": marker_plan[marker],
            "normalized_value": np.nan,
            "normalization_reference": "vehicle_within_model_pair_repeat",
            "qc_status": "pending",
            "operator_note": "",
        })
    pd.DataFrame(mechanism_rows).to_csv(
        OUT / "mechanism_assay_results_template.tsv", sep="\t", index=False
    )

    dictionary = [
        ["raw_plate_results", "plate_id", "string", "required", "must match the locked 36-plate manifest"],
        ["raw_plate_results", "well", "string", "required", "A1-H12; unique within plate"],
        ["raw_plate_results", "raw_signal", "float", "required for analysis", "untransformed instrument readout; non-negative"],
        ["raw_plate_results", "read_status", "category", "required", "pending, measured, or excluded"],
        ["raw_plate_results", "exclusion_reason", "string", "conditional", "required only when read_status is excluded"],
        ["plate_metadata", "run_date", "ISO date", "required before analysis", "YYYY-MM-DD"],
        ["plate_metadata", "cell_seeding_density_per_well", "integer", "required before analysis", "positive integer"],
        ["plate_metadata", "assay_duration_hours", "float", "required before analysis", "positive value"],
        ["mechanism_assay", "normalized_value", "float", "required for mechanism gate", "vehicle-normalized assay value"],
    ]
    pd.DataFrame(dictionary, columns=["table", "field", "type", "requirement", "rule"]).to_csv(
        OUT / "ingestion_data_dictionary.tsv", sep="\t", index=False
    )

    lock_files = [
        "primary_plate_manifest.tsv", "locked_96well_layout.tsv", "primary_pair_lock.tsv",
        "primary_backup_model_lock.tsv", "combination_analysis_lock.tsv",
    ]
    contract = {
        "contract_version": "1.0.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_population": "all locked plates passing outcome-blind plate QC",
        "randomization_seed": RANDOMIZATION_SEED,
        "required_completed_files": {
            "raw_plate_results_template.tsv": "copy to completed_raw_plate_results.tsv and fill raw_signal/read_status",
            "plate_metadata_template.tsv": "copy to completed_plate_metadata.tsv and fill experimental metadata",
            "mechanism_assay_results_template.tsv": "copy to completed_mechanism_assay_results.tsv when mechanism assays are available",
        },
        "plate_qc": {
            "blank_roles": ["media_blank", "no_cell_blank"],
            "vehicle_role": "vehicle_control",
            "positive_kill_role": "positive_kill_control",
            "vehicle_cv_max": 0.20,
            "z_prime_min": 0.50,
            "failure_action": "exclude the complete plate before computing combination scores",
        },
        "normalization": "blank subtraction followed by division by the within-plate blank-subtracted vehicle median; retain raw and untruncated normalized values",
        "primary_estimand": "mean Bliss excess across the complete 5x5 non-zero grid after averaging matched technical replicates",
        "secondary_estimands": {
            "HSA": "implemented locally on the identical locked grid",
            "ZIP_and_Loewe": "export to a version-pinned SynergyFinder workflow; supportive only",
        },
        "model_positive_gate": [
            "median Bliss score > 10",
            "bootstrap 95% CI lower bound > 0",
            "positive Bliss direction in all three biological repeats",
        ],
        "pair_advance_gate": "positive in at least 3 of 4 core Hybrid models, including at least one HER2+ and one TNBC model",
        "lock_sha256": {name: sha256(LOCK / name) for name in lock_files},
        "outcomes_used_to_create_contract": False,
    }
    (OUT / "ingestion_and_analysis_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    readme = """# Hybrid combination result-ingestion contract

This directory contains prospective templates only; it contains no treatment-response result.

1. Copy `raw_plate_results_template.tsv` to `completed_raw_plate_results.tsv` and fill `raw_signal`, `read_status`, and any exclusions.
2. Copy `plate_metadata_template.tsv` to `completed_plate_metadata.tsv` and complete the experimental metadata.
3. Do not delete, add, or reorder locked plate/well identifiers.
4. Run `analyze_hybrid_combination_results.py --input-dir <completed-results-folder>` only after the templates are complete.
5. ZIP and Loewe are supportive outputs and must be produced with the locked SynergyFinder version from the exported long-format file.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "analysis_type": "prospective result-ingestion and synergy-analysis contract lock",
        "plates": len(manifest),
        "raw_template_rows": len(raw),
        "mechanism_template_rows": len(mechanism_rows),
        "randomization_seed": RANDOMIZATION_SEED,
        "response_data_used": False,
        "primary_synergy_metric": "mean Bliss excess over locked 5x5 non-zero grid",
        "next_gate": "wet-lab data remain external; computational study may proceed to final evidence synthesis",
    }
    (OUT / "result_ingestion_contract_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
