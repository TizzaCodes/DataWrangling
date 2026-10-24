# DMI vs EMI — Data Wrangling Assignment

This cleaned project runs the existing **DMI vs Gaussian EMI** experiment using a
single entry point. It keeps the original entropy-based decision-tree DMI
implementation (`src/dmi.py`) and Gaussian EM numerical imputer (`src/emi.py`).
The separate C4.5 implementation and old generated run artifacts are not part
of this cleaned package.

## Setup

Use Python 3.10 or newer. From this folder, create and activate a virtual
environment if desired, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Run the study

```bash
python main.py
```

On the first run, `main.py` generates the missingness scenarios if the
manifest or any referenced masked-data/mask file is absent. It then runs the
DMI and EMI methods for up to 640 manifest experiments. The runner checkpoints
CSV results after each completed experiment and resumes from those checkpoints
when `python main.py` is run again.

The full study can take a long time. Keep the terminal open while it runs.

### Useful options

```bash
# Smoke test: one experiment, first 500 rows
python main.py --test

# Run the first 10 experiments
python main.py --limit 10

# Run one dataset only
python main.py --dataset adult
python main.py --dataset credit_approval

# Start from scratch rather than resuming prior checkpoints
python main.py --no-resume

# Save imputed datasets as well as metrics (uses much more disk space)
python main.py --save-imputed
```

Combine options as needed. For example:

```bash
python main.py --dataset adult --limit 320
```

## Inputs and outputs

- `data/raw/`: original raw datasets.
- `data/processed/`: cleaned datasets used by the experiments.
- `data/missingness/`: generated automatically on the first run; contains the
  manifest and masked data/mask files for each experiment.
- `results/dmi/full_suite/`: output metrics and resumable checkpoints.

Important result files:

- `per_attribute_metrics.csv`: numeric imputation metrics by attribute.
- `pooled_metrics.csv`: pooled metrics per experiment and method.
- `experiment_diagnostics.csv`: runtime, missing-value checks, and method
  diagnostics.
- `summary_by_condition.csv`: metrics grouped by dataset, missingness pattern,
  ratio, distribution model, and method.
- `summary_by_attribute.csv`: metrics grouped by attribute and condition.

## Research limitation

This is the project's current implementation of the paper-inspired DMI
workflow, not a claim of exact reproduction of every paper detail. In
particular, the tree implementation uses scikit-learn's entropy-based decision
tree classifier rather than C4.5, and the EMI component is a Gaussian EM
implementation for numeric attributes. Report these implementation choices
when discussing results.
