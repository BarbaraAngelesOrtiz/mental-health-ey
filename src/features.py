# Mental Health Hackathon - Feature Engineering
# Author: Barbara Ortiz
# Purpose:
#   - Align dataset with Top 10 correlated features
#   - Apply semantic-safe transformations
#   - Prepare modeling-ready dataset
#   - Generate correlation diagnostics

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_features.csv")
CORR_OUTPUT = os.path.join(BASE_DIR, "data", "analysis", "feature_correlations.txt")
IMG_OUTPUT = os.path.join(BASE_DIR, "images", "Correlation_top_features_vs_target.png")

# LOAD DATA

df = pd.read_csv(INPUT_PATH)
print("Loaded cleaned dataset:", df.shape)

# TARGETS

TARGET_CURRENT = "Do you currently have a mental health disorder?"
TARGET_SEEK = "Have you ever sought treatment for a mental health issue from a mental health professional?"

df["target_current_condition"] = df[TARGET_CURRENT].astype(float)
df["target_seek_treatment"] = df[TARGET_SEEK].astype(float)

# SEMANTIC TRANSFORMATIONS

DIAGNOSIS_TEXT_COL = "If so, what condition(s) were you diagnosed with?"

if DIAGNOSIS_TEXT_COL in df.columns:
    df["diagnosis_reported_flag"] = df[DIAGNOSIS_TEXT_COL].notna().astype(int)

BINARY_COLS = [
    "Have you had a mental health disorder in the past?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "Do you have a family history of mental illness?",
]

for col in BINARY_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# TOP 10 FEATURES (BASED ON PEARSON ANALYSIS)

TOP10_CURRENT = [
    "Have you had a mental health disorder in the past?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "diagnosis_reported_flag",
    "Do you believe your productivity is ever affected by a mental health issue?",
    "Do you have a family history of mental illness?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Did you feel that your previous employers took mental health as seriously as physical health?",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?",
    "How willing would you be to share with friends and family that you have a mental illness?",
    "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?"
]

TOP10_SEEK = [
    "Have you had a mental health disorder in the past?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "diagnosis_reported_flag",
    "Do you believe your productivity is ever affected by a mental health issue?",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Do you have a family history of mental illness?",
    "How willing would you be to share with friends and family that you have a mental illness?",
    "Did you feel that your previous employers took mental health as seriously as physical health?",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"
]

# CORRELATION CHECK (SANITY CHECK)

df_current = df.dropna(subset=["target_current_condition"])
df_seek = df.dropna(subset=["target_seek_treatment"])

corr_current = (
    df_current[TOP10_CURRENT]
    .corrwith(df_current["target_current_condition"])
    .abs()
    .sort_values(ascending=False)
)

corr_seek = (
    df_seek[TOP10_SEEK]
    .corrwith(df_seek["target_seek_treatment"])
    .abs()
    .sort_values(ascending=False)
)

# Save correlation report
with open(CORR_OUTPUT, "w", encoding="utf-8") as f:
    f.write("Top 10 Correlations - Current Mental Health Condition\n")
    f.write(corr_current.to_string())
    f.write("\n\nTop 10 Correlations - Seeking Treatment\n")
    f.write(corr_seek.to_string())

# SAVE FINAL DATASET

# ORGANIZATIONAL COMPOSITE INDEXES

# Mental Health Support Index
support_components = [
    "Does your employer provide mental health benefits as part of healthcare coverage?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
    "Did you feel that your previous employers took mental health as seriously as physical health?"
]

support_components = [c for c in support_components if c in df.columns]

df["mental_health_support_index"] = df[support_components].mean(axis=1)

# Workplace Stigma Index
if "How willing would you be to share with friends and family that you have a mental illness?" in df.columns:
    df["willing_to_share_inverted"] = 1 - df[
        "How willing would you be to share with friends and family that you have a mental illness?"
    ]

stigma_components = [
    "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?",
    "willing_to_share_inverted"
]

stigma_components = [c for c in stigma_components if c in df.columns]

df["workplace_stigma_index"] = df[stigma_components].mean(axis=1)

# Organizational Openness Score
openness_components = [
    "How willing would you be to share with friends and family that you have a mental illness?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Did you feel that your previous employers took mental health as seriously as physical health?"
]

openness_components = [c for c in openness_components if c in df.columns]

df["organizational_openness_score"] = df[openness_components].mean(axis=1)

print("\nComposite organizational indexes created successfully.")

df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering completed successfully.")
print("Features saved to:", OUTPUT_PATH)
print("Correlation report saved to:", CORR_OUTPUT)

# GRAPHICAL OUTPUT

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
    "How willing would you be to share with friends and family that you have a mental illness?": "willing_to_share"
}

corr_plot = corr_current.rename(index=rename_dict)

plt.figure(figsize=(8, 5))
corr_plot.sort_values().plot(kind="barh")
plt.title("Top 10 Correlated Features – Current Condition")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig(IMG_OUTPUT, bbox_inches="tight")
plt.show()


