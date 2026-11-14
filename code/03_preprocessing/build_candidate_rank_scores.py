"""Create outcome-blind candidate signature scores with frozen development scaling."""
from __future__ import annotations

import os

import hashlib
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(os.environ.get("EMT_IMMUNE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEST = ROOT / "02_data/03_processed/analysis_ready_v1/candidate_rank_scores"
REPORT = ROOT / "04_analysis/02_preprocessing/candidate_rank_score_summary.json"
COMMON = pd.read_csv(ROOT / "02_data/02_interim/reference/cross_platform_gene_universe_v1/common_measured_gene_universe.tsv",
                     sep="\t")["gene_symbol"].astype(str).tolist()
RESPONSE_LOCKED_GENES = set(pd.read_csv(
    ROOT / "02_data/02_interim/treatment_response/probe_gene_lock_v1/locked_probe_to_gene_mapping.tsv",
    sep="\t")["gene_symbol"].astype(str))
SCORING_UNIVERSE = [g for g in COMMON if g in RESPONSE_LOCKED_GENES]
EMP = ROOT / "02_data/02_interim/externally_anchored_emp_v5.0_locked/signatures/locked_external_emp_signatures.tsv"
TIP = ROOT / "02_data/02_interim/reference/tip_cancer_immunity_cycle/TIP_cancer_immunity_cycle_28_signatures.gmt"
RESPONSE_EXCLUDE = {
    "TIP::Step3.negative", "TIP::Step4.CD4 T cell.recruiting", "TIP::Step4.Eosinophil.recruiting",
    "TIP::Step4.Basophil.recruiting", "TIP::Step6.negative", "TIP::Step7.negative",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def signatures() -> dict[str, set[str]]:
    e = pd.read_csv(EMP, sep="\t")
    out = {f"candidate_EMP::{k}": set(v["gene"].astype(str).str.upper()) for k, v in e.groupby("signature")}
    with TIP.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip().split("\t")
            if len(p) > 2:
                out[f"TIP::{p[0]}"] = {x.upper() for x in p[2:] if x}
    return out


def score_file(cohort: str, path: Path, allowed: list[str], sigs: dict[str, set[str]]) -> pd.DataFrame:
    with h5py.File(path, "r") as h:
        samples, genes = decode(h["sample_id"]), decode(h["gene_symbol"])
        gene_pos = {g: i for i, g in enumerate(genes)}
        missing_background = [g for g in SCORING_UNIVERSE if g not in gene_pos]
        if missing_background:
            raise ValueError(f"{cohort}: missing scoring-background genes: {missing_background[:20]}")
        common_idx = np.asarray([gene_pos[g] for g in SCORING_UNIVERSE], dtype=int)
        common_pos = {g: j for j, g in enumerate(SCORING_UNIVERSE)}
        sig_idx = {s: np.asarray([common_pos[g] for g in sorted(sigs[s]) if g in common_pos], dtype=int)
                   for s in allowed}
        x = h["X"]
        scores = np.empty((len(samples), len(allowed)), dtype=np.float64)
        for i in range(len(samples)):
            values = np.asarray(x[i, common_idx], dtype=np.float64)
            ranks = rankdata(values, method="average") / len(values)
            for j, s in enumerate(allowed):
                scores[i, j] = ranks[sig_idx[s]].mean() - 0.5
    return pd.DataFrame(scores, index=pd.Index(samples, name="sample_id"), columns=allowed)


def standardize(frames: dict[str, pd.DataFrame], development: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    mean = frames[development].mean(axis=0)
    sd = frames[development].std(axis=0, ddof=0)
    if (sd <= 0).any():
        raise ValueError(f"Zero-variance development scores: {sd.index[sd <= 0].tolist()}")
    params = pd.DataFrame({"feature": mean.index, "development_mean": mean.values, "development_sd": sd.values})
    return {k: (v - mean) / sd for k, v in frames.items()}, params


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    failure_file = REPORT.parent / "candidate_rank_score_failure.json"
    if failure_file.exists():
        failure_file.unlink()
    scoring_path = ROOT / "02_data/02_interim/reference/cross_platform_gene_universe_v1/scoring_gene_universe_v1.tsv"
    pd.Series(SCORING_UNIVERSE, name="gene_symbol").to_csv(scoring_path, sep="\t", index=False)
    sigs = signatures()
    survival_features = sorted(sigs)
    response_features = sorted(set(sigs) - RESPONSE_EXCLUDE)
    files = ROOT / "02_data/03_processed/analysis_ready_v1"
    survival = {c: score_file(c, files / "bulk_survival" / f"{c}_expression.h5", survival_features, sigs)
                for c in ["METABRIC", "TCGA_BRCA", "SCANB_GSE96058"]}
    response = {c: score_file(c, files / "treatment_response" / f"{c}_expression.h5", response_features, sigs)
                for c in ["GSE25055", "GSE25065"]}
    survival_z, s_params = standardize(survival, "METABRIC")
    response_z, r_params = standardize(response, "GSE25055")
    outputs = []
    for branch, raw_frames, z_frames in [("survival", survival, survival_z), ("response", response, response_z)]:
        for cohort in raw_frames:
            raw_path = DEST / f"{cohort}_{branch}_rank_scores.tsv.gz"
            z_path = DEST / f"{cohort}_{branch}_rank_scores_development_z.tsv.gz"
            raw_frames[cohort].to_csv(raw_path, sep="\t", compression="gzip")
            z_frames[cohort].to_csv(z_path, sep="\t", compression="gzip")
            outputs.append({"branch": branch, "cohort": cohort, "samples": len(raw_frames[cohort]),
                            "features": raw_frames[cohort].shape[1], "raw_file": str(raw_path),
                            "raw_sha256": sha(raw_path), "z_file": str(z_path), "z_sha256": sha(z_path),
                            "nonfinite": int((~np.isfinite(z_frames[cohort].to_numpy())).sum())})
    sparam_path, rparam_path = DEST / "survival_development_scaling.tsv", DEST / "response_development_scaling.tsv"
    s_params.to_csv(sparam_path, sep="\t", index=False)
    r_params.to_csv(rparam_path, sep="\t", index=False)
    summary = {"status": "PASS", "version": "1.0", "outcomes_used": False,
               "portability_universe_genes": len(COMMON),
               "rank_background_genes": len(SCORING_UNIVERSE),
               "rank_background_file": str(scoring_path), "rank_background_sha256": sha(scoring_path),
               "rank_background_rule": "common measured universe intersect unique GPL96 probe-to-gene lock",
               "survival_features": survival_features,
               "response_features": response_features, "response_excluded": sorted(RESPONSE_EXCLUDE),
               "outputs": outputs, "survival_scaling_sha256": sha(sparam_path),
               "response_scaling_sha256": sha(rparam_path),
               "next_gate": "candidate_representation_reliability_and_redundancy_qc"}
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs": outputs, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        (REPORT.parent / "candidate_rank_score_failure.json").write_text(
            json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
