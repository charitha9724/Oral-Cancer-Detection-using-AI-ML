"""
Tabular pipeline: trains and evaluates 6 models on the cleaned oral cancer dataset.

Models: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, MLP
Scaling: StandardScaler fit on train only, applied to all models.
Metrics: AUC-ROC, F1, Sensitivity, Specificity, MCC, Cohen's Kappa, Log Loss
Statistical comparison: McNemar's test on top model pairs.
"""

import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
from sklearn.utils.multiclass import unique_labels
from statsmodels.stats.contingency_tables import mcnemar

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_COL = "Oral Cancer (Diagnosis)"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv("tabular_train.csv")
    test = pd.read_csv("tabular_test.csv")
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].values
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].values
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


# ── Models ────────────────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_SEED, eval_metric="logloss",
            verbosity=0, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            max_iter=300, random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.1
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "AUC-ROC":      round(roc_auc_score(y_true, y_prob), 4),
        "F1":           round(f1_score(y_true, y_pred), 4),
        "Sensitivity":  round(sensitivity, 4),
        "Specificity":  round(specificity, 4),
        "MCC":          round(matthews_corrcoef(y_true, y_pred), 4),
        "Cohen Kappa":  round(cohen_kappa_score(y_true, y_pred), 4),
        "Log Loss":     round(log_loss(y_true, y_prob), 4),
    }


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(y_true, preds_a, preds_b, name_a, name_b):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=False, correction=True)
    sig = "YES" if result.pvalue < 0.05 else "NO"
    print(f"  McNemar ({name_a} vs {name_b}): chi2={result.statistic:.3f}, p={result.pvalue:.4f} -> significant={sig}")
    return result.pvalue


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nScaling features (StandardScaler fit on train)...")
    X_train_s, X_test_s = scale(X_train, X_test)

    models = get_models()
    results = {}
    all_preds = {}

    print("\n" + "="*65)
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        all_preds[name] = y_pred
        for k, v in metrics.items():
            print(f"  {k:<15} {v}")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("RESULTS SUMMARY")
    print("="*65)
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    print(df_results.to_string())
    df_results.to_csv("tabular_results.csv")
    print("\nSaved: tabular_results.csv")

    # ── McNemar's test: top 3 by AUC-ROC ─────────────────────────────────────
    print("\n" + "="*65)
    print("MCNEMAR'S TEST (top 3 models by AUC-ROC)")
    print("="*65)
    top3 = df_results["AUC-ROC"].nlargest(3).index.tolist()
    pairs = [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2])]
    mcnemar_results = {}
    for a, b in pairs:
        p = mcnemar_test(y_test, all_preds[a], all_preds[b], a, b)
        mcnemar_results[f"{a} vs {b}"] = round(p, 4)

    with open("tabular_mcnemar.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    print("Saved: tabular_mcnemar.json")

    # ── Best model ────────────────────────────────────────────────────────────
    best = df_results["AUC-ROC"].idxmax()
    print(f"\nBest model by AUC-ROC: {best}  ({df_results.loc[best, 'AUC-ROC']})")


if __name__ == "__main__":
    main()
