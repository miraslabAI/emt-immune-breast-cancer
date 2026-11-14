# EMT-state immune-cycle bottlenecks in breast cancer

Reproducible analysis code accompanying the manuscript **“EMT state marks distinct immune-cycle bottlenecks across breast cancer ecosystems.”**

## Scope

This repository contains the locked, outcome-blind bulk, malignant single-cell, spatial, pathway, target and drug-prioritization workflows used in the paper. It also contains the prospective experimental contracts. These contracts define future tests; they are not evidence of combination efficacy or a validated treatment-predictive biomarker.

Patient-level and controlled source data are not redistributed. Obtain the public/controlled datasets from the sources listed in `DATA_AVAILABILITY.md`, place them under `02_data/`, and retain the directory structure expected by the scripts.

## Reproducible setup

The reference analysis used Python 3.10.20. Create an environment from `environment.yml` or install `requirements.txt`. CPU execution is supported; CUDA was used only where explicitly requested and is not required for most workflows.

By default, scripts resolve the repository root from their own location. To use another project root, set `EMT_IMMUNE_PROJECT_ROOT`.

Windows PowerShell:

```powershell
$env:EMT_IMMUNE_PROJECT_ROOT = "D:\path\to\repository"
python code\09_biological_analysis\build_tcga_emt_immune_landscape.py
```

Linux/macOS:

```bash
export EMT_IMMUNE_PROJECT_ROOT=/path/to/repository
python code/09_biological_analysis/build_tcga_emt_immune_landscape.py
```

See `RUN_ORDER.md` for the manuscript-aligned execution sequence and `CODE_MANIFEST.tsv` for checksums.

## Repository map

- `code/`: analysis and figure-generation scripts
- `protocols/`: frozen analysis specifications
- `results/`: aggregate, non-identifiable final tables
- `reproducibility/`: sanitized environment records
- `DATA_AVAILABILITY.md`: dataset identifiers and redistribution boundaries

## Important claim boundary

The repository supports the reported biological model and state-qualified experimental hypotheses. It does not establish clinical efficacy, treatment benefit, or predictive validity.

## License

## License

This software is released under the MIT License. See `LICENSE` for details.
