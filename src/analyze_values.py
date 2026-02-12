# Mental Health Hackathon - Pre-Feature Analysis
# Author: Barbara Ortiz
# Purpose:
#   - Analyze cleaned dataset before feature engineering
#   - Missing diagnostics
#   - Target distributions
#   - Leakage detection
#   - Top 10 correlated features

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_cleaned.csv")
MISSING_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "analysis", "missing_summary.csv")

TARGET_CURRENT_CONDITION = "Do you currently have a mental health disorder?"
TARGET_SEEK_TREATMENT = "Have you ever sought treatment for a mental health issue from a mental health professional?"

# LOAD DATA

df = pd.read_csv(INPUT_PATH)

# DATASET OVERVIEW

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print("Shape:", df.shape)
print("Total missing cells:", df.isna().sum().sum())
print("Overall missing ratio:", round(df.isna().mean().mean(), 4))

# MISSING VALUES SUMMARY

UNKNOWN_VALUES = [-1, -2]

missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_ratio": df.isna().mean() + df.isin(UNKNOWN_VALUES).mean(),
    "missing_count": df.isna().sum() + df.isin(UNKNOWN_VALUES).sum(),
    "dtype": df.dtypes.values
}).sort_values("missing_ratio", ascending=False)

missing_summary.to_csv(MISSING_SUMMARY_PATH, index=False)

print("\nTop 15 columns by missing ratio:")
print(missing_summary.head(15))

# TARGET DISTRIBUTIONS

def print_target_distribution(target):
    print("\n" + "=" * 80)
    print(f"DISTRIBUTION: {target}")
    print("=" * 80)
    if target not in df.columns:
        print("❌ Target not found.")
        return
    dist = df[target].value_counts(dropna=False)
    dist_ratio = df[target].value_counts(dropna=False, normalize=True)
    summary = pd.DataFrame({"count": dist, "ratio": dist_ratio})
    print(summary)

print_target_distribution(TARGET_CURRENT_CONDITION)
print_target_distribution(TARGET_SEEK_TREATMENT)

print("\n⚠️ NOTE:")
print(" 'Seeking Treatment' is conditional on having a mental health condition.")
print(" Global correlations may be misleading. Conditional analysis recommended.")

# LEAKAGE CHECK BETWEEN TARGETS

print("\n" + "=" * 80)
print("LEAKAGE CHECK BETWEEN TARGETS")
print("=" * 80)

if TARGET_CURRENT_CONDITION in df.columns and TARGET_SEEK_TREATMENT in df.columns:
    sub = df[[TARGET_CURRENT_CONDITION, TARGET_SEEK_TREATMENT]].dropna()
    if len(sub) > 0:
        corr = sub.corr().iloc[0,1]
        print(f"Correlation between targets: {corr:.3f}")
        if abs(corr) > 0.7:
            print("⚠️ Strong correlation detected — potential leakage risk.")
        elif abs(corr) > 0.4:
            print("⚠️ Moderate correlation — review modeling design.")
        else:
            print("✅ No severe leakage detected.")
    else:
        print("Not enough overlapping non-null rows for correlation.")
else:
    print("❌ One or both target columns missing.")

# TOP 10 FEATURES CORRELATED WITH EACH TARGET

print("\n" + "=" * 80)
print("TOP 10 FEATURES CORRELATED WITH TARGETS (pre-features)")
print("=" * 80)

df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Keep only numeric
df_numeric = df_encoded.select_dtypes(include=[np.number])

# Remove targets from feature pool
feature_cols = [
    col for col in df_numeric.columns
    if col not in [TARGET_CURRENT_CONDITION, TARGET_SEEK_TREATMENT]
]

print(f"\nTotal numeric candidate features: {len(feature_cols)}")

# PEARSON CORRELATION (baseline)

df_current = df_numeric.dropna(subset=[TARGET_CURRENT_CONDITION])
df_seek = df_numeric.dropna(subset=[TARGET_SEEK_TREATMENT])

corr_current = (
    df_current[feature_cols]
    .corrwith(df_current[TARGET_CURRENT_CONDITION])
    .abs()
    .sort_values(ascending=False)
)

corr_seek = (
    df_seek[feature_cols]
    .corrwith(df_seek[TARGET_SEEK_TREATMENT])
    .abs()
    .sort_values(ascending=False)
)

# PREPARE DATA PER TARGET (REMOVE NaN TARGETS)

# CURRENT CONDITION
df_current = df_numeric.dropna(subset=[TARGET_CURRENT_CONDITION]).copy()

X_current = df_current[feature_cols].fillna(
    df_current[feature_cols].median()
)

y_current = df_current[TARGET_CURRENT_CONDITION].astype(int)

print(f"\nRows available for Current Condition: {len(df_current)}")

# SEEK TREATMENT
df_seek = df_numeric.dropna(subset=[TARGET_SEEK_TREATMENT]).copy()

X_seek = df_seek[feature_cols].fillna(
    df_seek[feature_cols].median()
)

y_seek = df_seek[TARGET_SEEK_TREATMENT].astype(int)

print(f"Rows available for Seeking Treatment: {len(df_seek)}")

# MUTUAL INFORMATION

mi_current = mutual_info_classif(
    X_current,
    y_current,
    random_state=42
)

mi_seek = mutual_info_classif(
    X_seek,
    y_seek,
    random_state=42
)

mi_current_df = pd.DataFrame({
    "feature": feature_cols,
    "mi_score": mi_current
}).sort_values("mi_score", ascending=False)

mi_seek_df = pd.DataFrame({
    "feature": feature_cols,
    "mi_score": mi_seek
}).sort_values("mi_score", ascending=False)

# TOP 10

top10_current_corr = corr_current.head(10)
top10_seek_corr = corr_seek.head(10)

top10_current_mi = mi_current_df.head(10)
top10_seek_mi = mi_seek_df.head(10)

print("\n PEARSON TOP 10 (Current Condition) ")
print(top10_current_corr)

print("\n MUTUAL INFORMATION TOP 10 (Current Condition) ")
print(top10_current_mi)

print("\n PEARSON TOP 10 (Seeking Treatment) ")
print(top10_seek_corr)

print("\n MUTUAL INFORMATION TOP 10 (Seeking Treatment) ")
print(top10_seek_mi)


print("\nANALYSIS COMPLETED")


if len(feature_cols) < 10:
    print("\n⚠️ WARNING: Less than 10 numeric features available in dataset.")
    print("Consider encoding more categorical variables.")

top10_features_current = top10_current_corr.index.tolist()


corr_matrix_top10 = df_numeric[top10_features_current].corr().abs()

high_corr_pairs = []

for i in range(len(corr_matrix_top10.columns)):
    for j in range(i+1, len(corr_matrix_top10.columns)):
        if corr_matrix_top10.iloc[i, j] > 0.8:
            high_corr_pairs.append((
                corr_matrix_top10.columns[i],
                corr_matrix_top10.columns[j],
                corr_matrix_top10.iloc[i, j]
            ))

if high_corr_pairs:
    print("\n⚠️ High collinearity detected within Top 10 (Current Condition):")
    for pair in high_corr_pairs:
        print(pair)
else:
    print("\n✅ No severe collinearity inside Top 10 (Current Condition).")

# GRAPHICAL ANALYSIS 

def shorten_label(text, max_len=50):
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

fig, ax = plt.subplots(1, 2, figsize=(10,4))

df[TARGET_CURRENT_CONDITION].value_counts().plot(kind="bar", ax=ax[0], title="Current Mental Health Condition")
df[TARGET_SEEK_TREATMENT].value_counts().plot(kind="bar", ax=ax[1], title="Seeking Treatment")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "Target_distributions.png"), bbox_inches="tight")
plt.show()

top_missing = missing_summary.head(10).copy()
top_missing["short_label"] = top_missing["column"].apply(shorten_label)

plt.figure(figsize=(8,5))
plt.barh(top_missing["short_label"], top_missing["missing_ratio"])
plt.title("Top 10 Columns by Missing Ratio")
plt.xlabel("Missing Ratio")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "Missing_ratio.png"), bbox_inches="tight")

plt.show()

