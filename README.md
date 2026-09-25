# LoanGuard AI: Loan Default Prediction System

LoanGuard AI is a professional, banking-themed web application developed in Streamlit for predicting loan default risk. The system employs a supervised machine learning binary classification model (Logistic Regression) to analyze applicant profiles and estimate default probability.

This project is tailored for academic demonstrations (e.g., B.Tech Computer Science ML projects and vivas), showcasing a production-ready model pipeline from training and serialization to a web-based risk dashboard.

---

## Key Features

- **Dashboard**: High-level bank metrics (Total Applications, Default Rate, Model Status) and interactive Plotly distribution charts (Credit Score, Income, Loan Amount, Target class distribution).
- **Loan Prediction**: Interactive multi-section input form (Applicant Info, Financial Info, Loan Info) with robust inputs, client-side validation, risk probability meter, and risk categorizations.
- **Risk Analysis**: Live visualization of trained model coefficients with statistical log-odds interpretations.
- **Model Information**: Architectural breakdown of the preprocessing pipeline, mathematical formulations (sigmoid equations rendered in LaTeX), performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC), and an interactive confusion matrix heatmap.
- **About & Explorer**: Detailed metadata glossary of all 18 variables, and an interactive database explorer to browse, search, and filter historical records.

---

## File & Project Structure

```
loan_default_prediction/
│
├── app.py                     # Main application entry point & navigation config
├── train_model.py             # Script to preprocess, train, and serialize model/artifacts
├── requirements.txt           # Python dependency specifications
├── README.md                  # Project setup and user guide
│
├── pages/                     # Separate page modules
│   ├── dashboard.py           # Dashboard landing page
│   ├── prediction.py          # Prediction form and result card
│   ├── risk_analysis.py       # Coefficient display and explanations
│   ├── model_info.py          # Mathematical concepts and performance metrics
│   └── about.py               # About project and Interactive Dataset Explorer
│
├── model/                     # Folder containing serialized artifacts (generated on train)
│   ├── logistic_regression_model.pkl  # Pickled sklearn Logistic Regression model
│   ├── scaler.pkl                     # Pickled sklearn StandardScaler object
│   ├── encoders.pkl                   # Pickled dictionary of LabelEncoder objects
│   └── metrics.json                   # Saved performance metrics and coefficients
│
├── data/                      # Data storage folder
│   ├── Loan_default.csv       # Original raw dataset (255,347 records)
│   └── Loan_default_cleaned.csv  # Preprocessed, encoded, and scaled dataset
│
├── utils/                     # Preprocessing and prediction scripts
│   ├── preprocessing.py       # Cleans, encodes, and scales user inputs
│   └── prediction.py          # Loads model and outputs probability predictions
│
└── assets/                    # Presentation assets
    └── custom.css             # Banking custom style sheets
```

---

## Setup & Running Instructions

Follow these steps to run the application on your local machine:

### 1. Prerequisite
Ensure that `Loan_default.csv` is placed in the root directory.

### 2. Create and Activate a Virtual Environment
It is highly recommended to isolate dependencies inside a virtual environment:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Train the Model and Preprocessors
Run the training script to clean the data, train the model, fit the scaler and encoders, and cache them inside the `model/` folder:

```bash
python train_model.py
```
This script will output the classification report and save the required `.pkl` files and `metrics.json`.

### 5. Run the Streamlit Application
Execute the Streamlit server to open the web portal in your default browser:

```bash
streamlit run app.py
```
If it doesn't open automatically, navigate to `http://localhost:8501` in your browser.

---

## Machine Learning Pipeline

1. **Outlier Filtering**: income values outside $Q_1 - 1.5 \times \text{IQR}$ and $Q_3 + 1.5 \times \text{IQR}$ are removed to handle extremes.
2. **Label Encoding**: Categorical fields are converted to alphabetical index representations (`0, 1, 2, ...`) via `LabelEncoder`.
3. **Standardization**: Continuous features are transformed using `StandardScaler` to have zero mean and unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
4. **Logistic Regression Classifier**: Log-odds are calculated using trained weights, and a sigmoid function maps outputs to default probabilities:
   $$P(\text{Default} = 1) = \frac{1}{1 + e^{-z}}$$
   $$z = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n$$

---

## Model Performance Summary

- **Accuracy**: 88.59%
- **Precision**: 62.16%
- **Recall**: 3.12%
- **ROC-AUC**: 0.7498
- **Confusion Matrix**:
  - True Negatives (TN): 45,058
  - False Positives (FP): 112
  - False Negatives (FN): 5,716
  - True Positives (TP): 184
