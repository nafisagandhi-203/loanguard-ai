import os
import json
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root():
    """Absolute path to the project root (one level above utils/)."""
    return PROJECT_ROOT


def get_possible_dataset_paths():
    """All candidate locations for the raw loan dataset."""
    return [
        os.path.join(PROJECT_ROOT, "Loan_default.csv"),
        os.path.join(PROJECT_ROOT, "Loan_Default.csv"),
        os.path.join(PROJECT_ROOT, "ipynb files", "Loan_default.csv"),
        os.path.join(PROJECT_ROOT, "ipynb files", "Loan_Default.csv"),
    ]


def get_dataset_path():
    """Return the first existing dataset path, or None if unavailable."""
    for path in get_possible_dataset_paths():
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    """
    Locate and load the raw loan dataset, cached for the session.
    Returns a DataFrame, or None if the dataset cannot be loaded.
    """
    data_path = get_dataset_path()
    if data_path is None:
        return None
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def get_model_metrics_path():
    """Absolute path to the serialized Logistic Regression metrics file."""
    return os.path.join(PROJECT_ROOT, "model", "metrics.json")


@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Load model/metrics.json (coefficients + evaluation metrics), or None."""
    metrics_path = get_model_metrics_path()
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None