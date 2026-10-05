import pandas as pd
import os

print("=" * 60)
print("H1B VISA APPROVAL PREDICTION")
print("=" * 60)

dataset_folder = "dataset"

files = os.listdir(dataset_folder)

print("\nFiles in dataset folder:")
for file in files:
    print(" -", file)

csv_files = [file for file in files if file.lower().endswith(".csv")]

if not csv_files:
    print("\nERROR: No CSV file found.")
    print("Please put h1b_kaggle.csv inside the dataset folder.")
    exit()

file_path = os.path.join(dataset_folder, csv_files[0])

print("\nLoading dataset:")
print(file_path)

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

if "CASE_STATUS" in df.columns:
    print("\nCASE STATUS:")
    print(df["CASE_STATUS"].value_counts())

print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)