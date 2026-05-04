from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -------------------------------
# Utility Functions
# -------------------------------
def validate_positive_int(value: str) -> int:
    integer_value = int(value)
    if integer_value < 1:
        raise argparse.ArgumentTypeError("Must be positive integer")
    return integer_value


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame, target_column: str = "math score"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encoding
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X.astype(float), y


# -------------------------------
# Main Pipeline
# -------------------------------
def run_pipeline(data_path: Path, n_components: int = 5):

    df = load_data(data_path)
    X, y = preprocess_data(df)

    print("Encoded Features:\n")
    print(X.head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining data size:", X_train.shape)
    print("Testing data size:", X_test.shape)

    # Ensure PCA components do not exceed available features
    n_components = min(n_components, X_train.shape[1])

    # PCA Pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=42)
    )

    # Fit PCA
    X_train_pca = pipeline.fit_transform(X_train)
    X_test_pca = pipeline.transform(X_test)

    pca = pipeline.named_steps["pca"]

    print("\n===== PCA RESULTS =====")
    print("Original features:", X.shape[1])
    print("PCA components:", n_components)
    print("Training shape:", X_train_pca.shape)
    print("Testing shape:", X_test_pca.shape)
    print("Explained variance:", sum(pca.explained_variance_ratio_))

    # -------------------------------
    # Linear Regression (FIX ADDED)
    # -------------------------------
    model = LinearRegression()
    model.fit(X_train_pca, y_train)

    print("\nModel trained successfully!")

    # Predictions
    y_pred = model.predict(X_test_pca)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n===== MODEL PERFORMANCE =====")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)

    # Save metrics
    output_path = Path(__file__).resolve().parent / "models" / "student_model.joblib"
    metrics_path = output_path.parent / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(metrics_path, "w") as f:
        json.dump({
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "features": int(n_components)
        }, f)

    # -------------------------------
    # Demo Prediction (VERY IMPORTANT)
    # -------------------------------
    print("\n--- New Student Prediction ---")

    sample = X_test.iloc[[0]]
    sample_pca = pipeline.transform(sample)

    pred = model.predict(sample_pca)

    print("Predicted Math Score:", pred[0])

    # -------------------------------
    # Save model and training columns
    # -------------------------------
    joblib.dump((pipeline, model, list(X.columns)), output_path)

    print("\nSaved model to:", output_path)


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "students.csv")
    parser.add_argument("--components", type=validate_positive_int, default=14)
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        run_pipeline(args.data, args.components)
        print("\nPipeline completed successfully!")
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()