import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "05_smart_agriculture_crop_digital_twin.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Crop_Type",
    "Soil_Moisture",
    "Soil_pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Irrigation_Hours",
    "Pest_Index",
    "Previous_Yield",
]
CATEGORICAL_COLUMNS = ["Crop_Type"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
LABEL_MAP = {"Healthy": 0, "Moderate": 1, "Poor": 2}


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    if df.isnull().sum().any():
        for column in df.columns:
            if column in CATEGORICAL_COLUMNS:
                df[column] = df[column].fillna(df[column].mode().iloc[0])
            else:
                df[column] = df[column].fillna(df[column].median())

    X = df[FEATURE_COLUMNS]
    y_class = df["Crop_Health"].map(LABEL_MAP)
    y_reg = df["Predicted_Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    classifier_models = {
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=10),
    }

    classifier_results = {}
    for name, model in classifier_models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="macro", zero_division=0)
        recall = recall_score(y_test, predictions, average="macro", zero_division=0)
        f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist()
        classifier_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "macro_f1": f1,
            "confusion_matrix": cm,
            "pipeline": pipeline,
        }

    best_classifier_name = max(
        classifier_results,
        key=lambda name: classifier_results[name]["macro_f1"],
    )
    best_classifier = classifier_results[best_classifier_name]["pipeline"]

    regression_results = {}
    for name, model in regression_models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_reg_train, y_reg_train)
        predictions = pipeline.predict(X_reg_test)
        mae = mean_absolute_error(y_reg_test, predictions)
        rmse = mean_squared_error(y_reg_test, predictions) ** 0.5
        r2 = r2_score(y_reg_test, predictions)
        regression_results[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pipeline": pipeline,
        }

    best_regressor_name = max(
        regression_results,
        key=lambda name: regression_results[name]["r2"],
    )
    best_regressor = regression_results[best_regressor_name]["pipeline"]

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": FEATURE_COLUMNS,
        "classification_metric": "Macro F1-score",
        "regression_metric": "R² score",
        "best_classifier": best_classifier_name,
        "best_regressor": best_regressor_name,
        "classification_report": {
            name: {
                "accuracy": round(results["accuracy"], 4),
                "precision": round(results["precision"], 4),
                "recall": round(results["recall"], 4),
                "macro_f1": round(results["macro_f1"], 4),
            }
            for name, results in classifier_results.items()
        },
        "regression_report": {
            name: {
                "mae": round(results["mae"], 4),
                "rmse": round(results["rmse"], 4),
                "r2": round(results["r2"], 4),
            }
            for name, results in regression_results.items()
        },
        "confusion_matrix": classifier_results[best_classifier_name]["confusion_matrix"],
        "label_map": LABEL_MAP,
    }

    joblib.dump(best_classifier, ARTIFACTS_DIR / "best_classifier_pipeline.joblib")
    joblib.dump(best_regressor, ARTIFACTS_DIR / "best_regressor_pipeline.joblib")
    (ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("BEST CLASSIFIER:", best_classifier_name)
    print(json.dumps(metadata["classification_report"][best_classifier_name], indent=2))
    print("BEST REGRESSOR:", best_regressor_name)
    print(json.dumps(metadata["regression_report"][best_regressor_name], indent=2))
