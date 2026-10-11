import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import math


def train_model():
    print("Loading Diabetes dataset...")
    data = load_diabetes()
    X, y = data.data, data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {data.feature_names}")

    # Save dataset as CSV
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv("diabetes_data.csv", index=False)
    print("Dataset saved as diabetes_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {math.sqrt(mse):.2f}")
    print(f"R2   : {r2:.4f}")

    joblib.dump(model, "diabetes_model.pkl")
    print("\nModel saved as diabetes_model.pkl")

    metrics = {
        "r2": float(r2),
        "mse": float(mse),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return r2


if __name__ == "__main__":
    train_model()
