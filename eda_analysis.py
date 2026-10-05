import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# H1B VISA APPROVAL PREDICTION
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 70)
print("H1B VISA DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_PATH = "dataset/h1b_kaggle.csv"
EDA_FOLDER = "eda"

os.makedirs(EDA_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ------------------------------------------------------------
# 4. Missing value analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percentage
})

print(missing_report)

missing_report.to_csv(
    os.path.join(EDA_FOLDER, "missing_values.csv")
)

# ------------------------------------------------------------
# 5. Duplicate analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DUPLICATE ANALYSIS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# ------------------------------------------------------------
# 6. Case status analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. CASE STATUS ANALYSIS")
print("=" * 70)

case_status = df["CASE_STATUS"].value_counts()

print(case_status)

case_status.to_csv(
    os.path.join(EDA_FOLDER, "case_status_counts.csv")
)

plt.figure(figsize=(12, 7))

case_status.plot(kind="bar")

plt.title("H1B Case Status Distribution")
plt.xlabel("Case Status")
plt.ylabel("Number of Applications")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "case_status_distribution.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 7. Year analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. YEAR ANALYSIS")
print("=" * 70)

year_counts = df["YEAR"].value_counts().sort_index()

print(year_counts)

year_counts.to_csv(
    os.path.join(EDA_FOLDER, "year_counts.csv")
)

plt.figure(figsize=(10, 6))

year_counts.plot(kind="bar")

plt.title("H1B Applications by Year")
plt.xlabel("Year")
plt.ylabel("Number of Applications")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "applications_by_year.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 8. Top employers
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. TOP EMPLOYERS")
print("=" * 70)

top_employers = (
    df["EMPLOYER_NAME"]
    .dropna()
    .value_counts()
    .head(15)
)

print(top_employers)

top_employers.to_csv(
    os.path.join(EDA_FOLDER, "top_employers.csv")
)

plt.figure(figsize=(12, 7))

top_employers.sort_values().plot(kind="barh")

plt.title("Top 15 H1B Employers")
plt.xlabel("Number of Applications")
plt.ylabel("Employer")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "top_employers.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 9. Top job titles
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. TOP JOB TITLES")
print("=" * 70)

top_jobs = (
    df["JOB_TITLE"]
    .dropna()
    .value_counts()
    .head(15)
)

print(top_jobs)

top_jobs.to_csv(
    os.path.join(EDA_FOLDER, "top_job_titles.csv")
)

plt.figure(figsize=(12, 7))

top_jobs.sort_values().plot(kind="barh")

plt.title("Top 15 H1B Job Titles")
plt.xlabel("Number of Applications")
plt.ylabel("Job Title")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "top_job_titles.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 10. Full-time vs part-time
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. FULL-TIME POSITION ANALYSIS")
print("=" * 70)

full_time = df["FULL_TIME_POSITION"].value_counts(dropna=False)

print(full_time)

full_time.to_csv(
    os.path.join(EDA_FOLDER, "full_time_position.csv")
)

plt.figure(figsize=(8, 6))

full_time.plot(kind="bar")

plt.title("Full-Time vs Part-Time H1B Positions")
plt.xlabel("Position Type")
plt.ylabel("Number of Applications")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "full_time_position.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 11. Wage analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. PREVAILING WAGE ANALYSIS")
print("=" * 70)

print(df["PREVAILING_WAGE"].describe())

print("\nMedian wage:")
print(df["PREVAILING_WAGE"].median())

plt.figure(figsize=(10, 6))

sns.histplot(
    df["PREVAILING_WAGE"].dropna(),
    bins=50
)

plt.xlim(
    0,
    df["PREVAILING_WAGE"].quantile(0.99)
)

plt.title("Distribution of Prevailing Wage")
plt.xlabel("Prevailing Wage (USD)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "wage_distribution.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 12. Worksite analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10. TOP WORKSITES")
print("=" * 70)

top_worksites = (
    df["WORKSITE"]
    .dropna()
    .value_counts()
    .head(15)
)

print(top_worksites)

top_worksites.to_csv(
    os.path.join(EDA_FOLDER, "top_worksites.csv")
)

plt.figure(figsize=(12, 7))

top_worksites.sort_values().plot(kind="barh")

plt.title("Top 15 H1B Worksites")
plt.xlabel("Number of Applications")
plt.ylabel("Worksite")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "top_worksites.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("\nGenerated files:")

for file in os.listdir(EDA_FOLDER):
    print(" -", file)

print("\nAll exploratory analysis completed successfully!")