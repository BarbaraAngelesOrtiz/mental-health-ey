# Mental Health Hackathon - Cleaning & ETL Pipeline
# Author: Barbara Ortiz
# Purpose:
#   - Governed ETL for survey data
#   - Schema-driven cleaning
#   - Semantic-safe transformations
#   - Automatic audits (pre / post)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "mental_health.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_cleaned.csv")

RAW_VALUE_REPORT = os.path.join(BASE_DIR, "data", "analysis", "value_report_raw.txt")
CLEAN_VALUE_REPORT = os.path.join(BASE_DIR, "data", "analysis", "value_report_clean.txt")

UNIQUE_COMPARISON_PATH = os.path.join(BASE_DIR, "data", "analysis", "unique_value_comparison.csv")
UNIQUE_PLOT_PATH = os.path.join(BASE_DIR, "images", "Unique_value_reduction.png")

def detect_unmapped_columns(df_clean):
    print("\n🔎 Checking for columns not governed by SCHEMA...\n")

    # Columnas que siguen siendo object
    object_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    # Columnas definidas en el schema
    schema_cols = [c.strip() for c in SCHEMA.keys()]

    unmapped = []

    for col in object_cols:
        if col.strip() not in schema_cols:
            unmapped.append(col)

    if unmapped:
        print("⚠️ Columns not governed by SCHEMA:")
        for col in unmapped:
            print(f" - {col}")
    else:
        print("✅ All object columns are governed by SCHEMA")

    return unmapped

def generate_unique_comparison(df_raw, df_clean):

    comparison = []

    for col in df_clean.columns:
        if col in df_raw.columns:

            raw_unique = df_raw[col].nunique(dropna=True)
            clean_unique = df_clean[col].nunique(dropna=True)

            comparison.append({
                "column": col,
                "unique_raw": raw_unique,
                "unique_clean": clean_unique,
                "reduction": raw_unique - clean_unique
            })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values("reduction", ascending=False)

    comp_df.to_csv(UNIQUE_COMPARISON_PATH, index=False)

    # Plot Top 10 reductions
    top10 = comp_df.head(10)

    plt.figure(figsize=(10,6))

    x = np.arange(len(top10))

    plt.bar(x - 0.2, top10["unique_raw"], width=0.4, label="Raw")
    plt.bar(x + 0.2, top10["unique_clean"], width=0.4, label="Clean")

    plt.xticks(x, top10["column"], rotation=75, ha="right")
    plt.ylabel("Number of Unique Values")
    plt.title("Semantic Noise Reduction - Top 10 reductions(Raw vs Clean)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(UNIQUE_PLOT_PATH, bbox_inches="tight")
    plt.close()

    print("Unique value comparison generated.")

# GENERIC TEXT NORMALIZATION

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ["", "nan", "none", "null", "n/a", "na", "prefer not to say"]:
        return np.nan
    return x

# MAPPERS

def map_binary(x):

    if isinstance(x, (int, float)) and x in [0, 1]:
        return int(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if x.startswith("yes"):
        return 1
    if x.startswith("no"):
        return 0

    return np.nan

def map_yn_unsure(x):

    if isinstance(x, (int, float)) and x in [0, 0.5, 1]:
        return float(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if x.startswith("yes"):
        return 1.0

    if (
        "unsure" in x
        or "not sure" in x
        or "don't know" in x
        or "dont know" in x
        or "maybe" in x
    ):
        return 0.5

    if x.startswith("no"):
        return 0.0

    if "not applicable" in x:
        return np.nan

    return np.nan

def map_frequency(x):

    if isinstance(x, (int, float)) and x in [1, 2, 3, 4, 5]:
        return int(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if x.startswith("never"):
        return 1
    if x.startswith("rarely"):
        return 2
    if x.startswith("sometimes"):
        return 3
    if x.startswith("often"):
        return 4
    if x.startswith("always"):
        return 5

    return np.nan

def map_likert_5(x):

    if isinstance(x, (int, float)) and x in [1, 2, 3, 4, 5]:
        return int(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if x.startswith("very difficult"):
        return 1
    if x.startswith("somewhat difficult"):
        return 2
    if x.startswith("neither"):
        return 3
    if x.startswith("somewhat easy"):
        return 4
    if x.startswith("very easy"):
        return 5

    return np.nan

def map_openness(x):

    if isinstance(x, (int, float)) and x in [0, 1, 2, 3, 4]:
        return int(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if "not open at all" in x:
        return 0
    if "somewhat not open" in x:
        return 1
    if "neutral" in x:
        return 2
    if "somewhat open" in x:
        return 3
    if "very open" in x:
        return 4

    return np.nan

def clean_company_size(x):
    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if "1-5" in x:
        return 1
    if "6-25" in x:
        return 2
    if "26-100" in x:
        return 3
    if "100-500" in x:
        return 4
    if "500-1000" in x:
        return 5
    if "more than 1000" in x or "1000+" in x:
        return 6
    return np.nan

def clean_age(x):
    try:
        age = int(float(x))
        if 15 <= age <= 90:
            return age
    except:
        pass
    return np.nan

def map_percentage(x):
    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if "0%" in x:
        return 0
    if "1-25" in x:
        return 1
    if "26-50" in x:
        return 2
    if "51-75" in x:
        return 3
    if "76-100" in x:
        return 4

    return np.nan

def map_negative_impact(x):

    if isinstance(x, (int, float)) and x in [0, 0.5, 1]:
        return float(x)

    x = normalize_text(x)
    if pd.isna(x):
        return np.nan

    if x.startswith("yes"):
        return 1.0

    if (
        "unsure" in x
        or "not sure" in x
        or "don't know" in x
        or "dont know" in x
        or "maybe" in x
    ):
        return 0.5

    if x.startswith("no"):
        return 0.0

    return np.nan

# SCHEMA CONTRACT

SCHEMA = {

    # Demographics
  
    "What is your age?": clean_age,
    "What is your gender?": normalize_text,
    "What country do you live in?": normalize_text,
    "What country do you work in?": normalize_text,
    "What US state or territory do you live in?": normalize_text,
    "What US state or territory do you work in?": normalize_text,
    "Which of the following best describes your work position?": normalize_text,

    # Binary (true yes/no)
    
    "Are you self-employed?": map_binary,
    "Do you have previous employers?": map_binary,
    "Is your employer primarily a tech company/organization?": map_binary,
    "Is your primary role within your company related to tech/IT?": map_binary,
    "Do you currently have a mental health disorder?": map_binary,
    "Have you had a mental health disorder in the past?": map_binary,
    "Have you been diagnosed with a mental health condition by a medical professional?": map_binary,
    "Have you ever sought treatment for a mental health issue from a mental health professional?": map_binary,
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?": map_binary,
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?": map_binary,

    # Frequency scale
    
    "Do you work remotely?": map_frequency,
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?": map_frequency,
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?": map_frequency,

    # Likert 1-5
   
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": map_likert_5,

    # Yes / No / Unsure
    
    "Do you believe your productivity is ever affected by a mental health issue?": map_yn_unsure,
    "Do you feel that your employer takes mental health as seriously as physical health?": map_yn_unsure,
    "Did you feel that your previous employers took mental health as seriously as physical health?": map_yn_unsure,
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": map_yn_unsure,
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": map_yn_unsure,
    "Would you bring up a mental health issue with a potential employer in an interview?": map_yn_unsure,
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?": map_yn_unsure,
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": map_yn_unsure,
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": map_yn_unsure,
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": map_yn_unsure,
    "Does your employer provide mental health benefits as part of healthcare coverage?": map_yn_unsure,
    "Do you know the options for mental health care available under your employer-provided coverage?": map_yn_unsure,
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": map_yn_unsure,
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": map_yn_unsure,
    "Do you have a family history of mental illness?": map_yn_unsure,
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": map_yn_unsure,

    "Do you know local or online resources to seek help for a mental health disorder?": map_yn_unsure,
    "Have your previous employers provided mental health benefits?": map_yn_unsure,
    "Were you aware of the options for mental health care provided by your previous employers?": map_yn_unsure,
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?": map_yn_unsure,
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?": map_yn_unsure,
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?": map_yn_unsure,
    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?": map_yn_unsure,
    "Do you think that discussing a physical health issue with previous employers would have negative consequences?": map_yn_unsure,
    "Would you have been willing to discuss a mental health issue with your previous co-workers?": map_yn_unsure,
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?": map_yn_unsure,
    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?": map_yn_unsure,
    "Do you feel that being identified as a person with a mental health issue would hurt your career?": map_yn_unsure,
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?": map_yn_unsure,
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?": map_yn_unsure,
    "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?": map_yn_unsure,

    # Openness scale (0–4)

    "How willing would you be to share with friends and family that you have a mental illness?": map_openness,

    # Negative impact perception

    "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?": map_negative_impact,
    "If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?": map_negative_impact,
    
    # Clean company size
    "How many employees does your company or organization have?": clean_company_size,

    # Map percentage
    "If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?": map_percentage,

}

# AUDIT FUNCTIONS

def semantic_audit(df_raw, df_clean):
    errors = []

    for schema_col in SCHEMA.keys():
        for df_col in df_clean.columns:
            
            if df_col.strip() == schema_col.strip():

                original_non_null = df_raw[df_col].notna().sum()
                cleaned_non_null = df_clean[df_col].notna().sum()

                if original_non_null > 0 and cleaned_non_null == 0:
                    errors.append(f"{df_col}: LOST ALL INFORMATION DURING CLEANING")

    if errors:
        raise ValueError("SEMANTIC AUDIT FAILED:\n" + "\n".join(errors))

def generate_value_report(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for col in df.columns:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"COLUMN: {col}\n")
            f.write("=" * 80 + "\n")

            total = len(df)
            missing = df[col].isna().sum()

            f.write(f"Total rows: {total}\n")
            f.write(f"Missing: {missing} ({missing/total:.2%})\n\n")

            vc = df[col].value_counts(dropna=False)

            for v, c in vc.items():
                f.write(f"{c:5d} | {v}\n")

# MAIN ETL

def main():

    print("Loading raw dataset")
    df_raw = pd.read_csv(INPUT_PATH)

    print("Loading raw dataset")

    df_raw = pd.read_csv(INPUT_PATH)

    df_raw.columns = (
    df_raw.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u2011", "-", regex=False)
    )

    df_raw.columns = df_raw.columns.str.replace("\u2011", "-", regex=False)

    print("Applying schema-driven transformations")
    df_clean = df_raw.copy()

    for schema_col, mapper in SCHEMA.items():
        for df_col in df_clean.columns:
            if df_col.strip().lower() == schema_col.strip().lower():
                if mapper is not None:
                    df_clean[df_col] = df_clean[df_col].apply(mapper)

    print("Running semantic audit")
    semantic_audit(df_raw, df_clean)

    print("Saving cleaned dataset")
    df_clean.to_csv(OUTPUT_PATH, index=False)

    detect_unmapped_columns(df_clean)

    print("Generating reports")
    generate_value_report(df_raw, RAW_VALUE_REPORT)
    generate_value_report(df_clean, CLEAN_VALUE_REPORT)

    print("Generating unique value comparison")
    generate_unique_comparison(df_raw, df_clean)

    print("Cleaning completed successfully")

if __name__ == "__main__":
    main()
