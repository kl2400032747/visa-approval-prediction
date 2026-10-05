import pandas as pd
import os

# ============================================================
# H1B VISA PREDICTION
# FEATURE ENGINEERING
# ============================================================

INPUT_PATH = "dataset/h1b_cleaned.csv"
OUTPUT_PATH = "dataset/h1b_model_data.csv"

print("=" * 70)
print("H1B VISA FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load cleaned dataset
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Create binary target
# ------------------------------------------------------------

print("\nCreating target variable...")

status_mapping = {
    "CERTIFIED": 1,
    "CERTIFIED-WITHDRAWN": 1,
    "DENIED": 0,
    "WITHDRAWN": 0
}

df["TARGET"] = df["CASE_STATUS"].map(status_mapping)

# Remove rows where target could not be created
df = df.dropna(subset=["TARGET"])

df["TARGET"] = df["TARGET"].astype(int)

print("\nTarget distribution:")
print(df["TARGET"].value_counts())

print("\nTarget percentages:")
print(df["TARGET"].value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 3. Remove original target column
# ------------------------------------------------------------

df = df.drop(columns=["CASE_STATUS"])

# ------------------------------------------------------------
# 4. Create wage-related features
# ------------------------------------------------------------

if "PREVAILING_WAGE" in df.columns:

    df["WAGE_LOG"] = (
        df["PREVAILING_WAGE"]
        .clip(lower=0)
        .apply(lambda x: __import__("numpy").log1p(x))
    )

# ------------------------------------------------------------
# 5. Create year-based feature
# ------------------------------------------------------------

if "YEAR" in df.columns:

    df["YEAR"] = df["YEAR"].astype(int)

    df["YEARS_FROM_2011"] = df["YEAR"] - 2011

# ------------------------------------------------------------
# 6. Clean categorical features
# ------------------------------------------------------------

categorical_columns = [
    "EMPLOYER_NAME",
    "SOC_NAME",
    "JOB_TITLE",
    "FULL_TIME_POSITION",
    "WORKSITE"
]

for column in categorical_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

# ------------------------------------------------------------
# 7. Handle numeric missing values
# ------------------------------------------------------------

numeric_columns = [
    "PREVAILING_WAGE",
    "YEAR",
    "WAGE_LOG",
    "YEARS_FROM_2011"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(
            df[column].median()
        )

# ------------------------------------------------------------
# 8. Keep required features
# ------------------------------------------------------------

features = [
    "EMPLOYER_NAME",
    "SOC_NAME",
    "JOB_TITLE",
    "FULL_TIME_POSITION",
    "PREVAILING_WAGE",
    "YEAR",
    "WORKSITE",
    "WAGE_LOG",
    "YEARS_FROM_2011",
    "TARGET"
]

features = [
    column
    for column in features
    if column in df.columns
]

df = df[features]

# ------------------------------------------------------------
# 9. Remove duplicate records
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicate records removed:", before - after)

# ------------------------------------------------------------
# 10. Reset index
# ------------------------------------------------------------

df = df.reset_index(drop=True)

# ------------------------------------------------------------
# 11. Save model dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nModel dataset saved:")
print(OUTPUT_PATH)

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal target distribution:")
print(df["TARGET"].value_counts())

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)