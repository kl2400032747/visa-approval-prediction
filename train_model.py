import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# H1B VISA PREDICTION
# RANDOM FOREST MODEL
# ============================================================

DATA_PATH = "dataset/h1b_model_data.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("H1B VISA RANDOM FOREST MODEL")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading model dataset...")

df = pd.read_csv(DATA_PATH)

print("Full dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Stratified sampling
# ------------------------------------------------------------

SAMPLE_SIZE = 300_000

if len(df) > SAMPLE_SIZE:

    print(f"\nCreating stratified sample of {SAMPLE_SIZE:,} rows...")

    sample_parts = []

    for target_value, group in df.groupby("TARGET"):

        group_size = int(
            SAMPLE_SIZE *
            len(group) /
            len(df)
        )

        sample_parts.append(
            group.sample(
                n=group_size,
                random_state=42
            )
        )

    model_df = pd.concat(
        sample_parts
    ).sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

else:

    model_df = df.copy()

print("Training dataset shape:", model_df.shape)

print("\nTarget distribution:")
print(model_df["TARGET"].value_counts())

# ------------------------------------------------------------
# 3. Define features and target
# ------------------------------------------------------------

X = model_df.drop(columns=["TARGET"])
y = model_df["TARGET"]

# ------------------------------------------------------------
# 4. Define feature types
# ------------------------------------------------------------

categorical_features = [
    "EMPLOYER_NAME",
    "SOC_NAME",
    "JOB_TITLE",
    "FULL_TIME_POSITION",
    "WORKSITE"
]

numeric_features = [
    "PREVAILING_WAGE",
    "YEAR",
    "WAGE_LOG",
    "YEARS_FROM_2011"
]

# ------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------

print("\nPreparing categorical encoder...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ------------------------------------------------------------
# 7. Random Forest
# ------------------------------------------------------------

print("\nCreating Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# 8. Pipeline
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# ------------------------------------------------------------
# 9. Train
# ------------------------------------------------------------

print("\nTraining Random Forest...")
print("This may take several minutes.")

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed!")

# ------------------------------------------------------------
# 10. Prediction
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)

# ------------------------------------------------------------
# 11. Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ------------------------------------------------------------
# 12. Classification report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 13. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)

# ------------------------------------------------------------
# 14. Save model
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "visa_random_forest.pkl"
)

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:")

print(model_path)

# ------------------------------------------------------------
# 15. Save evaluation results
# ------------------------------------------------------------

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_rows": len(X_train),
    "testing_rows": len(X_test)
}

results_df = pd.DataFrame(
    [results]
)

results_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "model_metrics.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 16. Save confusion matrix
# ------------------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

cm_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.csv"
    )
)

print("\nEvaluation files saved.")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)