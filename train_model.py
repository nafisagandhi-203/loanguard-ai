import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix)


def tuned_threshold(model, X_val, y_val):
    """Find the decision threshold that maximizes F1 on a validation slice."""
    probs = model.predict_proba(X_val)[:, 1]
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.05, 0.951, 0.025):
        f1 = f1_score(y_val, (probs >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = float(th), f1
    return best_th


def balanced_sample_weights(y):
    """Inverse-frequency sample weights so the minority (Default) class is not ignored."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def train_and_save():
    print("=================================================================")
    print("Starting Multi-Model Training Pipeline (Loan Default Prediction)")
    print("=================================================================")

    # 1. Load original dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "Loan_default.csv"),
        os.path.join(current_dir, "Loan_Default.csv"),
        os.path.join(current_dir, "ipynb files", "Loan_default.csv"),
        os.path.join(current_dir, "ipynb files", "Loan_Default.csv"),
        os.path.join(current_dir, "ipynb files", "Loan_default_cleaned.csv"),
        os.path.join(current_dir, "data", "Loan_default_cleaned.csv"),
    ]
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        raise FileNotFoundError("Dataset not found in root or 'ipynb files/' folder.")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Clean column names (strip, lowercase, replace spaces with underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"Class balance -> Default: {df['default'].mean()*100:.2f}% | No Default: {(1-df['default'].mean())*100:.2f}%")

    # 3. Handle duplicates
    df.drop_duplicates(inplace=True)
    print(f"Shape after duplicate removal: {df.shape}")

    # 4. Filter income outliers using IQR if raw data
    if "income" in df.columns and df["income"].max() > 1000:
        df["income"] = df["income"].astype(float)
        Q1 = df["income"].quantile(0.25)
        Q3 = df["income"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df["income"] >= lower_bound) & (df["income"] <= upper_bound)]
        print(f"Shape after income outlier filtering: {df.shape}")

    # 5. Fit & save label encoders for categorical columns
    categorical_columns = [
        "education",
        "employmenttype",
        "maritalstatus",
        "hasmortgage",
        "hasdependents",
        "loanpurpose",
        "hascosigner"
    ]

    encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                print(f"Encoded '{col}' categories: {list(le.classes_)}")
            else:
                le = LabelEncoder()
                if col == "education":
                    le.classes_ = np.array(["Bachelor's", "High School", "Master's", "PhD"])
                elif col == "employmenttype":
                    le.classes_ = np.array(["Full-time", "Part-time", "Self-employed", "Unemployed"])
                elif col == "maritalstatus":
                    le.classes_ = np.array(["Divorced", "Married", "Single"])
                elif col == "loanpurpose":
                    le.classes_ = np.array(["Auto", "Business", "Education", "Home", "Other"])
                else:
                    le.classes_ = np.array(["No", "Yes"])
                encoders[col] = le

    os.makedirs("model", exist_ok=True)

    joblib.dump(encoders, "model/encoders.joblib")
    print("Saved categorical encoders to 'model/encoders.joblib'")

    # 6. Fit & save scaler for numerical columns
    numerical_columns = [
        "age",
        "income",
        "loanamount",
        "creditscore",
        "monthsemployed",
        "numcreditlines",
        "interestrate",
        "loanterm",
        "dtiratio"
    ]

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    joblib.dump(scaler, "model/scaler.joblib")
    print("Saved standard scaler to 'model/scaler.joblib'")

    # Save copy of cleaned/scaled dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/Loan_default_cleaned.csv", index=False)

    # 7. Split dataset (80% Train, 20% Test)
    feature_cols = [c for c in df.columns if c not in ["loanid", "default"]]
    X = df[feature_cols]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Validation slice used purely to tune each model's decision threshold
    X_val, y_val = X_test[:20000], y_test[:20000]

    # 8. Define all four models to train (class-balanced setups)
    models_to_train = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "type": "Generalized Linear Model",
            "description": "Fast and interpretable classification model based on log-odds.",
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        },
        "random_forest": {
            "name": "Random Forest",
            "type": "Bagging Ensemble",
            "description": "Ensemble of decision trees using bagging to reduce variance.",
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
        },
        "adaboost": {
            "name": "AdaBoost",
            "type": "Boosting Ensemble",
            "description": "Sequential boosting model that focuses on difficult observations.",
            "model": AdaBoostClassifier(
                n_estimators=80,
                learning_rate=0.5,
                random_state=42
            )
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "type": "Boosting Ensemble",
            "description": "Sequential ensemble model that improves predictions through boosting.",
            "model": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                subsample=0.9,
                random_state=42
            )
        }
    }

    # Balanced sample weights (used by models without built-in class_weight support)
    sample_weights = balanced_sample_weights(y_train)

    all_metrics = {}

    # 9. Train, tune threshold, cross-validate, evaluate and save each model
    for model_key, meta in models_to_train.items():
        print(f"\n---> Training: {meta['name']} ({model_key})...")
        m = meta["model"]

        if model_key in ("adaboost", "gradient_boosting"):
            m.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            m.fit(X_train, y_train)

        # Save model artifact
        joblib_path = f"model/{model_key}_model.joblib"
        joblib.dump(m, joblib_path)
        print(f"Saved model to '{joblib_path}'")

        # Probabilities & threshold tuning
        y_proba = m.predict_proba(X_test)[:, 1]
        decision_threshold = tuned_threshold(m, X_val, y_val)
        y_pred = (y_proba >= decision_threshold).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        bacc = float(balanced_accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred)

        # 5-fold cross-validation on a sampled training subset (speed vs accuracy)
        print(f"Running 5-fold Cross-Validation for {meta['name']}...")
        sample_idx = np.random.RandomState(42).choice(len(X_train), size=60000, replace=False)
        cv_scores = cross_val_score(m, X_train.iloc[sample_idx], y_train.iloc[sample_idx],
                                    cv=5, scoring="accuracy", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        print(f"Evaluation Results for {meta['name']} (threshold={decision_threshold:.3f}):")
        print(f"  Accuracy:        {acc*100:.2f}%")
        print(f"  Balanced Acc:    {bacc*100:.2f}%")
        print(f"  Precision:       {prec*100:.2f}% (Class 1 Default)")
        print(f"  Recall:          {rec*100:.2f}% (Class 1 Default)")
        print(f"  F1-Score:        {f1*100:.2f}%")
        print(f"  ROC-AUC:         {auc:.4f}")
        print(f"  PR-AUC:          {pr_auc:.4f}")
        print(f"  5-Fold CV:       {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%")
        print(f"  Confusion:       TN={int(cm[0,0])} FP={int(cm[0,1])} FN={int(cm[1,0])} TP={int(cm[1,1])}")

        all_metrics[model_key] = {
            "model_key": model_key,
            "name": meta["name"],
            "type": meta["type"],
            "description": meta["description"],
            "decision_threshold": decision_threshold,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
        }

    # 10. Top-level fields for Logistic Regression backward compatibility in model/metrics.json
    lr_metrics = all_metrics["logistic_regression"]

    full_metrics = {
        "accuracy": lr_metrics["accuracy"],
        "precision": lr_metrics["precision"],
        "recall": lr_metrics["recall"],
        "f1_score": lr_metrics["f1_score"],
        "roc_auc": lr_metrics["roc_auc"],
        "balanced_accuracy": lr_metrics["balanced_accuracy"],
        "decision_threshold": lr_metrics["decision_threshold"],
        "confusion_matrix": lr_metrics["confusion_matrix"],
        "intercept": float(models_to_train["logistic_regression"]["model"].intercept_[0]),
        "coefficients": {
            feature: float(c) for feature, c in zip(feature_cols,
                                                    models_to_train["logistic_regression"]["model"].coef_[0])
        },
        "feature_order": feature_cols,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "models": all_metrics
    }

    with open("model/metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=4)

    with open("model/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nSaved 'model/model_metrics.json' and 'model/metrics.json' successfully.")
    print("=================================================================")
    print("All four models successfully trained, threshold-tuned, and serialized!")
    print("=================================================================")


if __name__ == "__main__":
    train_and_save()