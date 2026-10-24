
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove records containing naturally missing values."""
    original_rows = len(df)
    clean_df = df.dropna().copy()
    removed_rows = original_rows - len(clean_df)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Original records: {original_rows}")
    print(f"Records removed:  {removed_rows}")
    print(f"Complete records: {len(clean_df)}")
    print(f"Remaining missing values: {clean_df.isna().sum().sum()}")

    return clean_df


# Adult dataset
adult_columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

adult = pd.read_csv(
    RAW_DIR / "adult.data",
    header=None,
    names=adult_columns,
    skipinitialspace=True,
    na_values="?"
)

# Credit Approval dataset
credit_columns = [f"A{i}" for i in range(1, 17)]

credit = pd.read_csv(
    RAW_DIR / "crx.data",
    header=None,
    names=credit_columns,
    skipinitialspace=True,
    na_values="?"
)

# Clean both datasets
adult_clean = clean_dataset(adult, "Adult")
credit_clean = clean_dataset(credit, "Credit Approval")

# Validate the counts reported in the paper
assert len(adult_clean) == 30162, (
    f"Adult count mismatch: expected 30162, got {len(adult_clean)}"
)
assert len(credit_clean) == 653, (
    f"Credit Approval count mismatch: expected 653, got {len(credit_clean)}"
)

# Save clean datasets
adult_clean.to_csv(PROCESSED_DIR / "adult_clean.csv", index=False)
credit_clean.to_csv(PROCESSED_DIR / "credit_clean.csv", index=False)

print("\nBoth clean datasets saved successfully.")
    