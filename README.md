# 🧠 Workplace Mental Health Risk Modeling
## EY Data Challenge – Predictive ML & Employee Profiling

---

# 🏆 Hackathon Context

This project was developed as part of the **EY Data Challenge Hackathon 2026**.

### 🎯 Use Case

Build an explainable Machine Learning system to predict mental health risk factors (e.g., anxiety, depression, stress) using structured survey data, and provide actionable insights for early detection and workplace intervention.

Deliverables required:

* Cleaned dataset
* Engineered indices
* Two ML models
* Clustering (3 employee profiles)
* Ranked feature importance
* Business recommendations
* Excel submission template completion

---

# 📊 Dataset

Anonymous survey of technology industry professionals, including:

* Current, past, diagnosed, treated mental health status
* Workplace productivity impact
* Company mental health benefits & policies
* Workplace culture (stigma & openness)
* Demographic and job-related data

---

# 🔄 Analytical Workflow

The solution follows a structured and auditable pipeline:

```
1. Raw Data Audit
2. Data Cleaning (Schema-driven ETL)
3. Post-cleaning validation
4. Feature Engineering (Index construction)
5. Correlation ranking
6. Clustering (3 profiles)
7. Supervised modeling (2 targets)
8. Interpretation & Business insights
```

Each stage was validated before moving forward to prevent leakage and ensure reproducibility.

---
# 🗂 Repository Structure

````
mental-health-ey/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── mental_health.csv
│   │   └── submission_template.xlsx
│   │
│   ├── processed/
│   │   ├── mental_health_cleaned.csv
│   │   ├── mental_health_features.csv
│   │   ├── model_condition_with_probs.csv
│   │   ├── model_treatment_with_probs.csv
│   │   └── submission_filled.xlsx
│   │
│   └── analysis/
│       ├── clusters.csv
│       ├── cluster_metrics.txt
│       ├── cluster_summary.csv
│       ├── correlation_matrix.csv
│       ├── feature_correlations.txt
│       ├── missing_summary.csv
│       ├── value_report_clean.txt
│       └── value_report_raw.txt
│
├── images/
│   ├── Cluster_distribution.png
│   ├── Cluster_profiles_zscore.png
│   ├── Correlation_top_features_vs_target.png
│   ├── Feature_correlation_matrix.png
│   ├── Missing_ratio.png
│   ├── PCA.png
│   ├── Radar_cluster_0.png
│   ├── Radar_cluster_1.png
│   ├── Radar_cluster_2.png
│   ├── Target_distributions.png
│   │
│   ├── condition/
│   │   ├── CM_logistic.png
│   │   ├── Probability_logistic.png
│   │   ├── PR_logistic.png
│   │   ├── Radar_logistic.png
│   │   ├── RF_feature_importance.png
│   │   └── ROC_logistic.png
│   │
│   └── treatment/
│       ├── CM_logistic.png
│       ├── Probability_logistic.png
│       ├── PR_logistic.png
│       ├── Radar_logistic.png
│       ├── RF_feature_importance.png
│       └── ROC_logistic.png
│
├── models/
│   ├── logistic_model_condition.pkl
│   ├── logistic_model_treatment.pkl
│   ├── logistic_threshold_condition.pkl
│   ├── logistic_threshold_treatment.pkl
│   ├── rf_model_condition.pkl
│   ├── rf_model_treatment.pkl
│   ├── rf_threshold_condition.pkl
│   └── rf_threshold_treatment.pkl
│
├──  src/
│    ├── analyze_values.py
│    ├── cleaning.py
│    ├── clustering.py
│    ├── features.py
│    ├── model_condition.py
│    ├── model_treatment.py
│    └── submission_excel.py 
│
└── requirements. txt 
````
---

## 🧹 1️. Data Preparation & Quality Assurance

✔ Cleaned and normalized categorical responses

✔ Handled missing values (“N/A”, “I don’t know”, “Prefer not to say”)

✔ Encoded binary, ordinal, and Likert scales

✔ Implemented schema-based transformations

✔ Generated raw and cleaned value audit reports

Output:

````
data/processed/mental_health_cleaned.csv
````

---

## 🧮 2️. Feature Engineering & Index Construction

Three composite indices were engineered as required:

---

### 🟢 Mental Health Support Index

Captures institutional support:

* Benefits availability
* Resource visibility
* Anonymity protection
* Formal communication

Top correlated field pairs were identified as required by the submission template.

---

### 🟠 Workplace Stigma Index

Captures perceived negative consequences:

* Fear of employer reaction
* Observed discrimination
* Client impact
* Disclosure hesitation

Top 5 correlation pairs provided.

---

### 🔵 Organizational Openness Score

Captures comfort discussing mental health:

* With coworkers
* With supervisors
* With family/friends

Top 5 correlation pairs provided.

All engineered features were saved into a modeling-ready dataset before clustering and supervised learning.

---

## 👥 3️. Clustering — Worker Profiling

Objective: Identify three distinct employee profiles.

KMeans was selected because:

* Higher silhouette score than Agglomerative
* Stability on standardized numeric features
* Clear interpretability
* Suitable for fixed k=3 requirement

---

### 🟠 Cluster 0 - Low Support / Low Openness

~63% of sample

* Many small companies
* Almost no mental health benefits
* Extremely low support index (~0.05)
* Stigma present

**Interpretation:**
Structurally under-supported environments.
Systemically vulnerable population.

---

### 🔵 Cluster 1 — Large Companies / Moderate Support

~16% of sample

* Formal policies exist
* Moderate support
* Lower stigma
* Openness similar to other clusters

**Key Insight:**
Having policies ≠ feeling psychologically safe.

---

### 🟢 Cluster 2 — High Benefits / Strong Support

~20% of sample

* Many small companies
* Benefits nearly universal
* Highest support index
* Slightly higher stigma than cluster 1

**Insight:**
Small organizations can outperform large enterprises in mental health culture.

Top 3 defining variables per cluster were identified and exported as required by submission format.

---

## 🤖 4️. Supervised Modeling

Two required targets:

### 🎯 Model 1

Target:
**“Do you currently have a mental health disorder?”**

* Logistic Regression ROC-AUC: **0.923**
* F1 Score ≈ 0.90
* Top 10 correlated features identified and ranked
* Model trained only on selected top features (as required)

---

### 🎯 Model 2

Target:
**“Have you ever sought treatment for a mental health issue from a mental health professional?”**

* Logistic Regression ROC-AUC: **0.922**
* F1 Score ≈ 0.85
* Top 10 correlated features identified
* Model trained using only those selected predictors

Random Forest was evaluated but Logistic Regression performed better and provided clearer interpretability.

---

## 🔎 Model Interpretability

Feature importance analysis performed via:

* Logistic coefficients
* Permutation importance
* Correlation ranking
* Mutual information

Strongest drivers:

* Past disorder
* Clinical diagnosis
* Diagnosis confirmation
* Family history
* Productivity impact
* Untreated interference

Diagnosis-related variables showed moderate collinearity (r ≈ 0.84), handled explicitly during evaluation.

---

## 📈 Key Business Insights

1. Diagnosis history is the strongest predictor of both condition and treatment.
2. Organizational support perception influences outcomes but does not dominate.
3. A large segment reports productivity impact without formal diagnosis.
4. Policy presence does not automatically create psychological safety.
5. Company size alone does not determine mental health support quality.

---

## 🏅 Hackathon Outcome Alignment

This solution delivers:

✔ An explainable ML model

✔ A ranked list of key workplace factors

✔ Clearly identified employee risk profiles

✔ Correlation pairs for each engineered index

✔ Cluster-defining features

✔ Concrete organizational recommendations

✔ Reproducible code pipeline

---

## 🛠 Technical Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* KMeans Clustering
* Logistic Regression
* Random Forest
* Silhouette Analysis

---

## 🎯 Final Reflection

This project demonstrates:

* Governed data engineering
* Leakage-aware modeling
* Structured feature selection
* Interpretable clustering
* Business-aligned analytics

It bridges technical rigor with organizational impact.

---

## 🚀 How to Run

### Central execution script

```bash
python notebooks/main.ipynb
```

#### 1️. Cleaning

```bash
python src/cleaning.py
```

#### 2️. Feature Engineering

```bash
python src/features.py
```

#### 3️. Clustering

```bash
python src/clustering.py
```

#### 4️. Modeling

```bash
python src/modeling_condition.py
python src/modeling_treatment.py
```
#### 5 Clustering

```bash
python src/submission_excel.py
```
---

## 📂 Outputs

Generated artifacts:

* Cleaned dataset
* Feature-engineered dataset
* Cluster metrics
* Model prediction files with probabilities
* Serialized thresholds and models

---

## 🛠️ Instructions for Running the Notebook

1. Clone or download this repository:

```bash
git clone https://github.com/user/mental-health-ey
```
2. Install the necessary dependencies (recommended: use a virtual environment):

```bash
pip install pandas matplotlib seaborn numpy plotly math matplotlib requests
```
3. Open the notebook in Jupyter, VSCode, or Google Colab:

4. Run the cells sequentially to replicate the full analysis.

---

## 📂 Project Access

- [Notebook](notebooks/main.ipynb)
- [Cleaning](src/cleaning.py)
- [Analyze](src/analyze_values.py)
- [Features](src/features.py)
- [Clustering](src/clustering.py)
- [Condition Model](src/model_condition.py)
- [Treatment Model](src/model_treatment.py)
- [Submission Excel](src/submission_excel.py)


---

## Author

**Bárbara Ángeles Ortiz**

<img src="https://github.com/user-attachments/assets/30ea0d40-a7a9-4b19-a835-c474b5cc50fb" width="115">

[LinkedIn](https://www.linkedin.com/in/barbaraangelesortiz/) | [GitHub](https://github.com/BarbaraAngelesOrtiz)

![Status](https://img.shields.io/badge/Status-Completed-success) 
![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)

![EY Data Challenge](https://img.shields.io/badge/EY-Data%20Challenge-yellow) 📅 February 2026

![ML Project](https://img.shields.io/badge/Machine%20Learning-Project-purple)
![Feature Engineering](https://img.shields.io/badge/Feature%20Engineering-Advanced-blueviolet)
![Clustering](https://img.shields.io/badge/Clustering-Worker%20Profiling-teal)
![Explainable AI](https://img.shields.io/badge/Explainable-AI-important)
![Data Cleaning](https://img.shields.io/badge/Data%20Cleaning-ETL-lightgrey)

![Python](https://img.shields.io/badge/python-3.10-blue)
![NumPy](https://img.shields.io/badge/numpy-1.26.0-blue)
![Pandas](https://img.shields.io/badge/pandas-2.1.0-blue)

![Matplotlib](https://img.shields.io/badge/matplotlib-3.8.0-blue)
![Seaborn](https://img.shields.io/badge/seaborn-0.13.0-blue)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)

![KMeans](https://img.shields.io/badge/Clustering-KMeans-green)
![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-success)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-success)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)

