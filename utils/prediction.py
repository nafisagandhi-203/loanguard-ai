import os
import json
import joblib
import streamlit as st

# Controlled Model Mapping
MODEL_MAP = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "type": "Generalized Linear Model",
        "filename": "logistic_regression_model.joblib",
        "fallback_filename": "logistic_regression_model.pkl",
        "description": "Fast and interpretable classification model based on log-odds.",
        "icon": "📈"
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "Bagging Ensemble",
        "filename": "random_forest_model.joblib",
        "fallback_filename": "random_forest_model.pkl",
        "description": "Ensemble of decision trees using bagging to reduce variance.",
        "icon": "🌲"
    },
    "adaboost": {
        "name": "AdaBoost",
        "type": "Boosting Ensemble",
        "filename": "adaboost_model.joblib",
        "fallback_filename": "adaboost_model.pkl",
        "description": "Sequential boosting model that focuses on difficult observations.",
        "icon": "⚡"
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "type": "Boosting Ensemble",
        "filename": "gradient_boosting_model.joblib",
        "fallback_filename": "gradient_boosting_model.pkl",
        "description": "Sequential ensemble model that improves predictions through boosting.",
        "icon": "🚀"
    }
}

# Risk Threshold Constants
RISK_THRESHOLDS = {
    "LOW_MAX": 0.30,      # 0.00 - 0.30: Low
    "MEDIUM_MAX": 0.60    # 0.30 - 0.60: Medium, > 0.60: High
}

def get_supported_models():
    """Returns the dictionary of all supported model metadata."""
    return MODEL_MAP

@st.cache_resource
def load_model(model_key="logistic_regression"):
    """
    Safely load the specified trained model from disk.
    Cached so it's loaded only once per session / process.
    Rejects any unmapped or arbitrary model identifier.
    """
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unsupported prediction model: '{model_key}'. Allowed: {list(MODEL_MAP.keys())}")
        
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    
    meta = MODEL_MAP[model_key]
    primary_path = os.path.join(project_root, "model", meta["filename"])
    fallback_path = os.path.join(project_root, "model", meta["fallback_filename"])
    
    if os.path.exists(primary_path):
        return joblib.load(primary_path)
    elif os.path.exists(fallback_path):
        return joblib.load(fallback_path)
    else:
        raise FileNotFoundError(
            f"Model artifact for '{meta['name']}' not found at '{primary_path}'. "
            "Please run 'python train_model.py' to generate all model files."
        )

@st.cache_data
def load_all_model_metrics():
    """Load evaluation metrics for all four models from model/model_metrics.json."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    metrics_path = os.path.join(project_root, "model", "model_metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

def calculate_risk_level(default_prob):
    """Derives standard risk category from default probability."""
    if default_prob <= RISK_THRESHOLDS["LOW_MAX"]:
        return "Low"
    elif default_prob <= RISK_THRESHOLDS["MEDIUM_MAX"]:
        return "Medium"
    else:
        return "High"


@st.cache_data
def get_decision_threshold(model_key="logistic_regression"):
    """
    Returns the tuned decision threshold for a model (from training metrics).
    The threshold was optimized to maximize F1 on a validation slice, so a
    probability above it counts as a predicted default. Falls back to 0.5.
    """
    try:
        metrics = load_all_model_metrics()
        model = metrics.get(model_key, {})
        if isinstance(model, dict) and "decision_threshold" in model:
            return float(model["decision_threshold"])
    except Exception:
        pass
    return 0.5

def predict_loan_risk(preprocessed_df, model_key="logistic_regression"):
    """
    Passes preprocessed features to the selected trained ML model
    and returns prediction class, probability of default, probability of no-default,
    and the evaluated risk level.
    """
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unsupported prediction model: '{model_key}'.")
        
    model = load_model(model_key)
    meta = MODEL_MAP[model_key]

    # 1. Deterministic class from tuned decision threshold (matches reported metrics)
    threshold = get_decision_threshold(model_key)
    default_prob = None

    # 2. Probability extraction
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(preprocessed_df)[0]
        no_default_prob = float(probabilities[0])
        default_prob = float(probabilities[1])
    else:
        default_prob = float(model.predict(preprocessed_df)[0])
        no_default_prob = 1.0 - default_prob

    predicted_class = int(default_prob >= threshold)

    # 3. Derive risk tier (probability-based, independent of decision threshold)
    risk_level = calculate_risk_level(default_prob)
    
    return {
        "success": True,
        "model": model_key,
        "model_name": meta["name"],
        "prediction": predicted_class,
        "prediction_label": "Default" if predicted_class == 1 else "No Default",
        "default_probability": default_prob,
        "no_default_probability": no_default_prob,
        "risk_level": risk_level,
        "decision_threshold": threshold
    }
