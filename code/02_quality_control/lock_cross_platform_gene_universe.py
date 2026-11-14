"""Lock the outcome-blind common measured gene universe and signature eligibility."""
from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
AUDIT = ROOT / "04_analysis/01_data_audit/expression_portability"
DEST = ROOT / "02_data/02_interim/reference/cross_platform_gene_universe_v1"
EMP = ROOT / "02_data/02_interim/externally_anchored_emp_v5.0_locked/signatures/locked_external_emp_signatures.tsv"
TIP = ROOT / "02_data/02_interim/reference/tip_cancer_immunity_cycle/TIP_cancer_immunity_cycle_28_signatures.gmt"


def load_universe(name: str) -> set[str]:
    return set(pd.read_csv(AUDIT / f"{name}_gene_universe.tsv", sep="\t")["gene_symbol"].astype(str).str.upper())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    names = ["METABRIC", "TCGA_BRCA", "SCANB_GSE96058", "GSE25055", "GSE25065"]
    universes = {n: load_universe(n) for n in names}
    common = set.intersection(*universes.values())
    cpath = DEST / "common_measured_gene_universe.tsv"
    pd.Series(sorted(common), name="gene_symbol").to_csv(cpath, sep="\t", index=False)

    e = pd.read_csv(EMP, sep="\t")
    sigs = {f"candidate_EMP::{k}": set(v["gene"].astype(str).str.upper()) for k, v in e.groupby("signature")}
    with TIP.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip().split("\t")
            if len(p) > 2:
                sigs[f"TIP::{p[0]}"] = {x.upper() for x in p[2:] if x}
    rows = []
    for sname, genes in sigs.items():
        record = {"signature": sname, "n_total": len(genes)}
        coverages = []
        for n in names:
            cov = len(genes & universes[n]) / len(genes)
            record[f"coverage_{n}"] = round(cov, 4)
            coverages.append(cov)
        record["minimum_coverage"] = round(min(coverages), 4)
        record["eligible_all_primary_cohorts"] = min(coverages) >= .80
        rows.append(record)
    spath = DEST / "candidate_signature_portability.tsv"
    pd.DataFrame(rows).to_csv(spath, sep="\t", index=False)
    summary = {"status": "PASS", "lock_version": "1.0", "outcomes_used": False,
               "cohort_gene_counts": {k: len(v) for k, v in universes.items()},
               "common_gene_count": len(common), "common_gene_file": str(cpath), "common_gene_sha256": sha(cpath),
               "signature_portability_file": str(spath), "signature_portability_sha256": sha(spath),
               "eligible_signatures": [r["signature"] for r in rows if r["eligible_all_primary_cohorts"]],
               "ineligible_signatures": [r["signature"] for r in rows if not r["eligible_all_primary_cohorts"]],
               "note": "Eligibility indicates measurability only, not biological or predictive validity.",
               "next_gate": "freeze_preprocessing_and_candidate_representation_registry"}
    (DEST / "cross_platform_gene_universe_lock.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
