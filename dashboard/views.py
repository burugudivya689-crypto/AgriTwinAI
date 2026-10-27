import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "05_smart_agriculture_crop_digital_twin.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_FILES = {
    "classifier": ARTIFACTS_DIR / "best_classifier_pipeline.joblib",
    "regressor": ARTIFACTS_DIR / "best_regressor_pipeline.joblib",
    "model_metadata": ARTIFACTS_DIR / "model_metadata.json",
}

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


def _load_model_bundle():
    if not MODEL_FILES["classifier"].exists() or not MODEL_FILES["regressor"].exists():
        raise FileNotFoundError("Models not found. Train the models first.")

    classifier = joblib.load(MODEL_FILES["classifier"])
    regressor = joblib.load(MODEL_FILES["regressor"])
    metadata = json.loads(MODEL_FILES["model_metadata"].read_text())
    return classifier, regressor, metadata


def _prepare_input(payload):
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")

        data = {key: payload.get(key) for key in FEATURE_COLUMNS}
        missing = [key for key, value in data.items() if value is None]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        df = pd.DataFrame([data])
        df["Crop_Type"] = df["Crop_Type"].astype(str)

        for col in [
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
        ]:
            df[col] = pd.to_numeric(df[col], errors="raise")

        return df
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _compute_prediction(df, classifier, regressor):
    classification = classifier.predict(df)
    yield_pred = regressor.predict(df)
    return classification[0], float(yield_pred[0])


def _format_health_label(label):
    label_map = {0: "Healthy", 1: "Moderate", 2: "Poor"}
    if isinstance(label, (list, np.ndarray)):
        label = label[0]
    return label_map.get(int(label), str(label))


def index(request):
    return render(request, "dashboard/index.html")


@csrf_exempt
def api_status(request):
    return JsonResponse({
        "status": "ok",
        "project": "AgriTwinAI",
        "dataset_exists": DATASET_PATH.exists(),
        "models_ready": MODEL_FILES["classifier"].exists() and MODEL_FILES["regressor"].exists(),
    })


@csrf_exempt
def api_options(request):
    crop_types = ["Maize", "Rice", "Cotton", "Groundnut", "Tomato"]
    return JsonResponse({
        "crop_types": crop_types,
        "classification_labels": ["Healthy", "Moderate", "Poor"],
        "health_map": {"Healthy": 0, "Moderate": 1, "Poor": 2},
        "notes": "This dashboard uses model-based predictions from the trained AgriTwinAI dataset. It is not a physical field simulation.",
    })


@csrf_exempt
def api_predict(request):
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        df = _prepare_input(payload)
        classifier, regressor, _ = _load_model_bundle()
        health, yield_pred = _compute_prediction(df, classifier, regressor)
        return JsonResponse({
            "success": True,
            "crop_health": _format_health_label(health),
            "predicted_yield": round(float(yield_pred), 3),
            "input_features": payload,
        })
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=400)


@csrf_exempt
def api_simulate(request):
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        baseline = payload.get("baseline") or payload
        modified = payload.get("modified") or payload
        if not isinstance(baseline, dict) or not isinstance(modified, dict):
            raise ValueError("Provide baseline and modified field inputs.")

        classifier, regressor, _ = _load_model_bundle()
        baseline_df = _prepare_input(baseline)
        modified_df = _prepare_input(modified)

        baseline_health, baseline_yield = _compute_prediction(baseline_df, classifier, regressor)
        modified_health, modified_yield = _compute_prediction(modified_df, classifier, regressor)

        return JsonResponse({
            "success": True,
            "note": "This is a model-based comparison only, not a guaranteed causal or physical simulation.",
            "baseline": {
                "crop_health": _format_health_label(baseline_health),
                "predicted_yield": round(float(baseline_yield), 3),
            },
            "modified": {
                "crop_health": _format_health_label(modified_health),
                "predicted_yield": round(float(modified_yield), 3),
            },
            "yield_change": round(float(modified_yield - baseline_yield), 3),
        })
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=400)


@csrf_exempt
def api_fields(request):
    if not DATASET_PATH.exists():
        return JsonResponse({"success": False, "error": "Dataset file not found."}, status=404)

    df = pd.read_csv(DATASET_PATH)
    recent = df.head(5).to_dict(orient="records")
    return JsonResponse({"success": True, "field_count": len(df), "sample_records": recent})


@csrf_exempt
def api_metrics(request):
    try:
        _, _, metadata = _load_model_bundle()
        return JsonResponse({
            "success": True,
            "best_classifier": metadata.get("best_classifier"),
            "best_regressor": metadata.get("best_regressor"),
            "classification_metric": metadata.get("classification_metric"),
            "regression_metric": metadata.get("regression_metric"),
            "classification_report": metadata.get("classification_report", {}),
            "regression_report": metadata.get("regression_report", {}),
            "confusion_matrix": metadata.get("confusion_matrix", []),
        })
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=404)
