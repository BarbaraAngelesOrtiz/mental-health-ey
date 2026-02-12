# Mental Health Hackathon - Model TREATMENT
# Author: Barbara Ortiz
# Purpose:
#   Predict likelihood of having sought mental health treatment
#   using organizational support, stigma, and workplace context.
#   Includes cross-validation, threshold tuning and model comparison.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve
)
import joblib
from sklearn.inspection import permutation_importance

def plot_confusion(y_test, y_pred, title, filename, image_dir):
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()


def plot_roc_curve(y_test, y_proba, title, filename, image_dir):
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()


def plot_pr_curve(y_test, y_proba, title, filename, image_dir):
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()


def plot_radar(metrics_dict, title, filename, image_dir):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()


def plot_probability_distribution(y_proba, title, filename, image_dir):
    plt.figure()
    plt.hist(y_proba, bins=30)
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()

def plot_feature_importance(
    model_pipeline,
    feature_names,
    title,
    filename,
    image_dir
):
    """
    Plots feature importance for tree-based models inside a pipeline.
    """

    model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(6,5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, filename))
    plt.show()
    plt.close()


# CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_features.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_treatment_with_probs.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "treatment"
IMAGE_DIR = os.path.join(BASE_DIR, "images", MODEL_NAME)
os.makedirs(IMAGE_DIR, exist_ok=True)

TARGET = "target_seek_treatment"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# LOAD DATA

df = pd.read_csv(INPUT_PATH)

rename_dict = {
    "Have you had a mental health disorder in the past?": "past_disorder",
    "Have you been diagnosed with a mental health condition by a medical professional?": "diagnosed",
    "diagnosis_reported_flag": "diagnosis_flag",
    "Do you believe your productivity is ever affected by a mental health issue?": "productivity_affected",
    "Do you have a family history of mental illness?": "family_history",
    "Do you know the options for mental health care available under your employer-provided coverage?": "knows_options",
    "Did you feel that your previous employers took mental health as seriously as physical health?": "prev_employer_support",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?": "untreated_interference",
    "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?": "client_negative_impact",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "medical_leave_difficulty",
    "How willing would you be to share with friends and family that you have a mental illness?": "willing_to_share"
}

df = df.rename(columns=rename_dict)

df_model = df.dropna(subset=[TARGET]).copy()
df_model[TARGET] = df_model[TARGET].astype(int)

# FEATURE SPACE 

feature_space = [
    "past_disorder",
    "diagnosed",
    "diagnosis_flag",
    "productivity_affected",
    "family_history",
    "untreated_interference",
    "knows_options",
    "prev_employer_support",
    "medical_leave_difficulty",
    "willing_to_share"
]

feature_space = [f for f in feature_space if f in df_model.columns]

X = df_model[feature_space]
y = df_model[TARGET]

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# CROSS VALIDATION

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# LOGISTIC REGRESSION

log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

log_pipeline.fit(X_train, y_train)

cv_scores_log = cross_val_score(
    log_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

y_proba_log = log_pipeline.predict_proba(X_test)[:, 1]

# Threshold tuning 
prec_log, rec_log, thresholds_log = precision_recall_curve(y_test, y_proba_log)
f1_scores_log = 2 * (prec_log * rec_log) / (prec_log + rec_log + 1e-9)
best_threshold_log = thresholds_log[np.argmax(f1_scores_log)]

y_pred_log = (y_proba_log >= best_threshold_log).astype(int)

print("\nLOGISTIC REGRESSION - TREATMENT")

cm_log = confusion_matrix(y_test, y_pred_log)
print("\nConfusion Matrix - Logistic Regression")
print(cm_log)

print("Best threshold:", round(best_threshold_log, 3))
print("CV ROC-AUC:", round(cv_scores_log.mean(), 3))
print(classification_report(y_test, y_pred_log))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba_log), 3))

coef_df = pd.DataFrame({
    "feature": feature_space,
    "coefficient": log_pipeline.named_steps["model"].coef_[0]
}).sort_values(by="coefficient", key=abs, ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df)

# RANDOM FOREST

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

rf_pipeline.fit(X_train, y_train)

result = permutation_importance(
    rf_pipeline,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE
)

perm_df = pd.DataFrame({
    "feature": feature_space,
    "importance": result.importances_mean
}).sort_values(by="importance", ascending=False)

print("\nPermutation Importance:")
print(perm_df)

cv_scores_rf = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Threshold tuning 
prec_rf, rec_rf, thresholds_rf = precision_recall_curve(y_test, y_proba_rf)
f1_scores_rf = 2 * (prec_rf * rec_rf) / (prec_rf + rec_rf + 1e-9)
best_threshold_rf = thresholds_rf[np.argmax(f1_scores_rf)]

y_pred_rf = (y_proba_rf >= best_threshold_rf).astype(int)

print("\nRANDOM FOREST - TREATMENT")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix - Random Forest")
print(cm_rf)

print("Best threshold:", round(best_threshold_rf, 3))
print("CV ROC-AUC:", round(cv_scores_rf.mean(), 3))
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba_rf), 3))

# SAVE PROBABILITIES

df_model["predicted_prob_treatment_log"] = (
    log_pipeline.predict_proba(X)[:, 1]
)

df_model["predicted_prob_treatment_rf"] = (
    rf_pipeline.predict_proba(X)[:, 1]
)

df_model.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Modeling completed successfully.")
print("Saved:", OUTPUT_PATH)

# MODELS SERIALIZED

joblib.dump(log_pipeline, os.path.join(MODEL_DIR, "logistic_model_treatment.pkl"))
joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, "rf_model_treatment.pkl"))

joblib.dump(best_threshold_log, os.path.join(MODEL_DIR, "logistic_threshold_treatment.pkl"))
joblib.dump(best_threshold_rf, os.path.join(MODEL_DIR, "rf_threshold_treatment.pkl"))

print("Models and thresholds serialized successfully.")

# PLOTS

# METRICS FOR RADAR CHART

metrics_log = {
    "ROC-AUC": roc_auc_score(y_test, y_proba_log),
    "Precision": precision_score(y_test, y_pred_log, average="macro"),
    "Recall": recall_score(y_test, y_pred_log, average="macro"),
    "F1": f1_score(y_test, y_pred_log, average="macro")
}

metrics_rf = {
    "ROC-AUC": roc_auc_score(y_test, y_proba_rf),
    "Precision": precision_score(y_test, y_pred_rf, average="macro"),
    "Recall": recall_score(y_test, y_pred_rf, average="macro"),
    "F1": f1_score(y_test, y_pred_rf, average="macro")
}

plot_confusion(
    y_test, y_pred_log,
    "Confusion Matrix - Logistic",
    "CM_logistic.png",
    IMAGE_DIR
)

plot_roc_curve(
    y_test, y_proba_log,
    "ROC Curve - Logistic",
    "ROC_logistic.png",
    IMAGE_DIR
)

plot_pr_curve(
    y_test, y_proba_log,
    "PR Curve - Logistic",
    "PR_logistic.png",
    IMAGE_DIR
)

metrics_log = {
    "ROC-AUC": roc_auc_score(y_test, y_proba_log),
    "Precision": precision_score(y_test, y_pred_log),
    "Recall": recall_score(y_test, y_pred_log),
    "F1": f1_score(y_test, y_pred_log)
}

plot_radar(
    metrics_log,
    "Performance Radar - Logistic",
    "Radar_logistic.png",
    IMAGE_DIR
)

plot_probability_distribution(
    y_proba_log,
    "Probability Distribution - Logistic",
    "Probability_logistic.png",
    IMAGE_DIR
)

plot_feature_importance(
    rf_pipeline,
    feature_space,
    f"Feature Importance - Random Forest ({MODEL_NAME.capitalize()})",
    "RF_feature_importance.png",
    IMAGE_DIR
)




