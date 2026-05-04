from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd


def predict_student(input_data: pd.DataFrame, model_path: Path):
    """Predict math score for new student"""

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run main.py first.")

    # Load saved pipeline + model
    pipeline, model = joblib.load(model_path)

    # Encode input (same as training)
    categorical_cols = input_data.select_dtypes(include=["object"]).columns.tolist()
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # IMPORTANT: match training columns
    # (this ensures same structure as training data)
    training_example = pd.read_csv(
        Path(__file__).resolve().parent / "data" / "students.csv"
    )
    X_train = training_example.drop(columns=["math score"])
    X_train = pd.get_dummies(X_train, drop_first=True)

    input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)

    # Apply PCA pipeline
    input_pca = pipeline.transform(input_encoded)

    # Predict
    prediction = model.predict(input_pca)

    return prediction


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "student_model.joblib"

    print("\n--- Predict New Student ---")

    # Example input
    new_student = pd.DataFrame({
        'gender': ['male'],
        'race/ethnicity': ['group C'],
        'parental level of education': ["bachelor's degree"],
        'lunch': ['standard'],
        'test preparation course': ['completed'],
        'reading score': [70],
        'writing score': [75]
    })

    try:
        prediction = predict_student(new_student, model_path)
        print("Predicted Math Score:", prediction[0])
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()