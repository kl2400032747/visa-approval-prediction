import pandas as pd
import os

# ============================================================
# H1B VISA PREDICTION
# DATA PREPROCESSING
# ============================================================

DATA_PATH = "dataset/h1b_kaggle.csv"
OUTPUT_PATH = "dataset/h1b_cleaned.csv"

print("=" * 70)
print("H1B VISA DATA PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Remove unnecessary index column
# ------------------------------------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nRemoved unnecessary index column.")

# ------------------------------------------------------------
# 3. Clean column names
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# 4. Clean categorical columns
# ------------------------------------------------------------

categorical_columns = [
    "CASE_STATUS",
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
            .astype("string")
            .str.strip()
            .str.upper()
        )

# ------------------------------------------------------------
# 5. Clean numerical columns
# ------------------------------------------------------------

if "PREVAILING_WAGE" in df.columns:

    df["PREVAILING_WAGE"] = pd.to_numeric(
        df["PREVAILING_WAGE"],
        errors="coerce"
    )

if "YEAR" in df.columns:

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce"
    )

if "lon" in df.columns:

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

if "lat" in df.columns:

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

# ------------------------------------------------------------
# 6. Remove rows without CASE_STATUS
# ------------------------------------------------------------

before = len(df)

df = df.dropna(subset=["CASE_STATUS"])

after = len(df)

print("\nRows removed because CASE_STATUS was missing:", before - after)

# ------------------------------------------------------------
# 7. Remove extremely rare / invalid statuses
# ------------------------------------------------------------

valid_statuses = [
    "CERTIFIED",
    "CERTIFIED-WITHDRAWN",
    "DENIED",
    "WITHDRAWN"
]

df = df[df["CASE_STATUS"].isin(valid_statuses)]

print("\nStatus distribution after cleaning:")

print(df["CASE_STATUS"].value_counts())

# ------------------------------------------------------------
# 8. Fill categorical missing values
# ------------------------------------------------------------

for column in [
    "EMPLOYER_NAME",
    "SOC_NAME",
    "JOB_TITLE",
    "FULL_TIME_POSITION"
]:

    if column in df.columns:

        df[column] = df[column].fillna("UNKNOWN")

# ------------------------------------------------------------
# 9. Fill numerical missing values
# ------------------------------------------------------------

if "PREVAILING_WAGE" in df.columns:

    df["PREVAILING_WAGE"] = (
        df["PREVAILING_WAGE"]
        .fillna(df["PREVAILING_WAGE"].median())
    )

if "YEAR" in df.columns:

    df["YEAR"] = (
        df["YEAR"]
        .fillna(df["YEAR"].median())
        .astype(int)
    )

# ------------------------------------------------------------
# 10. Handle longitude and latitude
# ------------------------------------------------------------

# We will not use longitude and latitude in the first model.
# They contain approximately 3.57% missing values.

if "lon" in df.columns:
    df = df.drop(columns=["lon"])

if "lat" in df.columns:
    df = df.drop(columns=["lat"])

# ------------------------------------------------------------
# 11. Remove duplicate rows
# ------------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    "\nDuplicate rows removed:",
    before_duplicates - after_duplicates
)

# ------------------------------------------------------------
# 12. Reset index
# ------------------------------------------------------------

df = df.reset_index(drop=True)

# ------------------------------------------------------------
# 13. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCleaned dataset saved to:")

print(OUTPUT_PATH)

print("\nFinal shape:")
print(df.shape)

print("\nRemaining missing values:")

print(df.isnull().sum())

print("\nFinal columns:")

print(df.columns.tolist())

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)