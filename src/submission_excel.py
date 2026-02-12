# Mental Health Hackathon - Submission Excel Filler - EY Hackathon
# Author: Barbara Ortiz
# Purpose: Fill official submission template

import os
import pandas as pd

# CONFIG

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEMPLATE_PATH = os.path.join(BASE_DIR, "data/raw/submission_template.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/submission_filled.xlsx")

# Load template

df = pd.read_excel(TEMPLATE_PATH)

row = {}

# ML TRAINING

# Model 1: Current Condition

condition_features = [
    "past_disorder",
    "diagnosed",
    "diagnosis_flag",
    "productivity_affected",
    "family_history",
    "knows_options",
    "prev_employer_support",
    "untreated_interference",
    "client_negative_impact",
    "willing_to_share"
]

for i, feat in enumerate(condition_features, start=1):
    row[f"Do you currently have a mental health disorder? corr {i}"] = feat

row["Do you currently have a mental health disorder? f1 score"] = 0.89

# Model 2: Treatment

treatment_features = [
    "past_disorder",
    "diagnosed",
    "diagnosis_flag",
    "productivity_affected",
    "family_history",
    "untreated_interference",
    "knows_options",
    "prev_employer_support",
    "client_negative_impact",
    "willing_to_share"
]

for i, feat in enumerate(treatment_features, start=1):
    row[
        "Have you ever sought treatment for a mental health issue from a mental health professional? corr " + str(i)
    ] = feat

row[
    "Have you ever sought treatment for a mental health issue from a mental health professional? f1 score"
] = 0.85


# FEATURE GROUPINGS (Support / Stigma / Openness)

# Mental Health Support
support_pairs = [
    ("knows_options", "prev_employer_support"),
    ("productivity_affected", "untreated_interference"),
    ("prev_employer_support", "knows_options"),
    ("family_history", "knows_options"),
    ("productivity_affected", "prev_employer_support"),
]

for i, (a, b) in enumerate(support_pairs, start=1):
    row[f"features mental health support {i}a"] = a
    row[f"features mental health support {i}b"] = b

# Workplace Stigma
stigma_pairs = [
    ("client_negative_impact", "willing_to_share"),
    ("untreated_interference", "client_negative_impact"),
    ("willing_to_share", "untreated_interference"),
    ("client_negative_impact", "diagnosed"),
    ("willing_to_share", "diagnosis_flag"),
]

for i, (a, b) in enumerate(stigma_pairs, start=1):
    row[f"features workplace stigma {i}a"] = a
    row[f"features workplace stigma {i}b"] = b

# Organizational Openness
openness_pairs = [
    ("willing_to_share", "prev_employer_support"),
    ("knows_options", "willing_to_share"),
    ("prev_employer_support", "knows_options"),
    ("productivity_affected", "willing_to_share"),
    ("prev_employer_support", "family_history"),
]

for i, (a, b) in enumerate(openness_pairs, start=1):
    row[f"features organizational openness {i}a"] = a
    row[f"features organizational openness {i}b"] = b

# CLUSTERING

clusters = {
    0: ["diagnosed", "diagnosis_flag", "past_disorder"],
    1: ["productivity_affected", "prev_employer_support", "untreated_interference"],
    2: ["prev_employer_support", "untreated_interference", "knows_options"]
}

for cluster_id, features in clusters.items():
    for j, feat in enumerate(features, start=1):
        row[f"cluster {cluster_id} {j}"] = feat

# Write row into template

for col, val in row.items():
    if col in df.columns:
        df.loc[0, col] = val

df.to_excel(OUTPUT_PATH, index=False)

print("✅ Submission Excel successfully generated:")
print(" ", OUTPUT_PATH)


