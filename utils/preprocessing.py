import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_preprocessors():
    """
    Load the standard scaler and label encoders from pickle files.
    Cached so they are loaded only once.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    scaler_joblib = os.path.join(project_root, "model", "scaler.joblib")
    scaler_pkl = os.path.join(project_root, "model", "scaler.pkl")
    scaler_path = scaler_joblib if os.path.exists(scaler_joblib) else scaler_pkl

    encoders_joblib = os.path.join(project_root, "model", "encoders.joblib")
    encoders_pkl = os.path.join(project_root, "model", "encoders.pkl")
    encoders_path = encoders_joblib if os.path.exists(encoders_joblib) else encoders_pkl
    
    if not os.path.exists(scaler_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError("Preprocessor artifacts (scaler or encoders) not found. Run model training first.")
        
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    return scaler, encoders

def preprocess_input(inputs_dict):
    """
    Takes a dictionary of raw inputs, applies scaling and label encoding,
    and returns a pandas DataFrame structured for the model in the correct feature order.
    """
    scaler, encoders = load_preprocessors()
    
    # 1. Standard feature order (must match X_train columns exactly)
    feature_order = [
        "age", "income", "loanamount", "creditscore", "monthsemployed", 
        "numcreditlines", "interestrate", "loanterm", "dtiratio",
        "education", "employmenttype", "maritalstatus", "hasmortgage", 
        "hasdependents", "loanpurpose", "hascosigner"
    ]
    
    # Create copies of numerical and categorical inputs
    processed = {}
    
    # 2. Preprocess numerical features
    num_cols = ["age", "income", "loanamount", "creditscore", "monthsemployed", "numcreditlines", "interestrate", "loanterm", "dtiratio"]
    num_vals_df = pd.DataFrame([[inputs_dict[col] for col in num_cols]], columns=num_cols)
    
    # Scale numerical values
    scaled_num_vals = scaler.transform(num_vals_df)[0]
    
    # Add scaled values to processed dict
    for i, col in enumerate(num_cols):
        processed[col] = scaled_num_vals[i]
        
    # 3. Preprocess categorical features
    # Mapping strings to integer values using fitted LabelEncoder objects
    cat_cols = ["education", "employmenttype", "maritalstatus", "hasmortgage", "hasdependents", "loanpurpose", "hascosigner"]
    for col in cat_cols:
        val = str(inputs_dict[col])
        try:
            # Map using the saved LabelEncoder
            encoded_val = encoders[col].transform([val])[0]
            processed[col] = int(encoded_val)
        except Exception as e:
            # Fallback mapping if there is an unseen label (though Streamlit selects limit options)
            # Default to index 0 of the label classes
            print(f"Error encoding {col} with value '{val}': {e}. Using fallback.")
            processed[col] = 0
            
    # 4. Construct DataFrame with exact column ordering
    features_df = pd.DataFrame([processed])[feature_order]
    
    return features_df
