"""
Run the project's DMI vs EMI study.

Default:
    python main.py

Quick smoke test (one experiment, first 500 rows):
    python main.py --test

Resume an interrupted study:
    python main.py

Run only one dataset:
    python main.py --dataset adult

The first run generates the 640 missingness scenarios if their files are absent.
The full DMI vs EMI study can take a long time; results are checkpointed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "missingness" / "manifest.csv"


def _resolve_manifest_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def ensure_missingness_data() -> None:
    """Create all missingness scenarios if the manifest/data are absent."""
    regenerate = not MANIFEST.exists()

    if not regenerate:
        try:
            manifest = pd.read_csv(MANIFEST)
            required = {"masked_data_file", "mask_file"}
            if not required.issubset(manifest.columns) or manifest.empty:
                regenerate = True
            else:
                # Verify the files required by the manifest exist before starting.
                for column in ("masked_data_file", "mask_file"):
                    if any(
                        not _resolve_manifest_path(value).exists()
                        for value in manifest[column].dropna()
                    ):
                        regenerate = True
                        break
        except Exception:
            regenerate = True

    if regenerate:
        print("=" * 72)
        print("Missingness files are absent or incomplete; generating them now.")
        print("This creates the study's 640 scenarios and their data/mask files.")
        print("=" * 72)
        from src.missingness_generator import generate_all
        generate_all()
    else:
        print(f"Using existing missingness manifest: {MANIFEST}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original entropy-tree DMI vs Gaussian EMI study."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run only the first experiment on the first 500 rows."
    )
    parser.add_argument(
        "--limit", type=int, default=640,
        help="Maximum number of manifest experiments to run (default: 640)."
    )
    parser.add_argument(
        "--rows", type=int, default=0,
        help="Rows per experiment; 0 means all rows (default: 0)."
    )
    parser.add_argument(
        "--dataset", choices=["adult", "credit_approval"],
        help="Optionally run only one dataset."
    )
    parser.add_argument(
        "--output-dir", default="results/dmi/full_suite",
        help="Where to save metrics and checkpoints."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of skipping completed experiments."
    )
    parser.add_argument(
        "--save-imputed", action="store_true",
        help="Save the completed data for every DMI and EMI experiment."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.limit < 1 or args.rows < 0:
        raise SystemExit("--limit must be positive and --rows non-negative.")

    if args.test:
        args.limit = 1
        args.rows = 500
        print("TEST MODE: one experiment, first 500 rows.")

    ensure_missingness_data()

    # Reuse the project's established, checkpointed experiment runner.
    from run_dmi_suite import main as run_suite

    runner_args = [
        "--limit", str(args.limit),
        "--rows", str(args.rows),
        "--output-dir", args.output_dir,
    ]
    if args.dataset:
        runner_args.extend(["--dataset", args.dataset])
    if not args.no_resume:
        runner_args.append("--resume")
    if args.save_imputed:
        runner_args.append("--save-imputed")

    original_argv = sys.argv[:]
    try:
        sys.argv = ["run_dmi_suite.py", *runner_args]
        return run_suite()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
