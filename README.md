# 🧠 Workplace Mental Health Risk Modeling
## Innovating Mental Health Risk Assessment: Predictive ML & Employee Profiling

---

## 🏆 Hackathon Context and Objective

This project was developed as part of the **EY Data Challenge Hackathon 2026: Innovating Mental Health Risk Assessment**. This project developed robust machine learning models to predict:

* The current presence of a mental health condition
* The likelihood of seeking professional treatment

Additionally, the analysis identified key structural risk drivers and segmented employees into actionable behavioral profiles to support targeted, data-driven organizational interventions.

---

## 🎯 Use Case

Build an explainable Machine Learning system to predict mental health risk factors (anxiety, depression, stress) using structured survey data, and provide actionable insights for early detection and workplace intervention.

Deliverables required:

* Cleaned dataset
* Engineered indices
* Two ML models
* Clustering (3 employee profiles)
* Ranked feature importance
* Business recommendations
* Excel submission template completion

---

## 📊 Dataset

* 1,433 technology professionals
* 69 engineered features
* 26.3% structured missingness (logic-based and conservative imputation applied)

Prevalence rates:

* 52% reported a current mental health condition
* 58.5% had sought professional treatment

A strong correlation was observed between condition presence and treatment-seeking behavior (r = 0.64). However, treatment is not universal, suggesting structural barriers such as limited awareness of support resources and perceived stigma.

Anonymous survey of technology industry professionals, including:

* Current, past, diagnosed, treated mental health status
* Workplace productivity impact
* Company mental health benefits & policies
* Workplace culture (stigma & openness)
* Demographic and job-related data

---

## 🔄 Analytical Workflow

The solution follows a structured and auditable pipeline:

```
1. Raw Data Audit
2. Data Cleaning (Schema-driven ETL)
3. Post-cleaning validation
4. Feature Engineering (Index construction)
5. Correlation ranking
6. Clustering (3 profiles)
7. Supervised modeling (2 targets)
8. Model Serialization
9. Interpretation & Business insights
```

Each stage was validated before moving forward to prevent leakage and ensure reproducibility.

---
## 🗂 Repository Structure

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
│   │   ├── SHAP_summary_condition.png
│   │   └── ROC_logistic.png
│   │
│   └── treatment/
│       ├── CM_logistic.png
│       ├── Probability_logistic.png
│       ├── PR_logistic.png
│       ├── Radar_logistic.png
│       ├── RF_feature_importance.png
│       ├── SHAP_summary_treatment.png
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
├── src/
│    ├── analyze_values.py
│    ├── cleaning.py
│    ├── clustering.py
│    ├── features.py
│    ├── model_condition.py
│    ├── model_treatment.py
│    └── submission_excel.py 
│ 
├── docs/
│    ├── Executive_summary.pdf
│    └── Report.pdf
│
└── requirements. txt 
````
---

## 🧹 Data Preparation & Quality Assurance

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

## 🧮 Feature Engineering & Index Construction

Three composite indices were engineered as required:

### 🟢 Mental Health Support Index

Captures institutional support:

* Benefits availability
* Resource visibility
* Anonymity protection
* Formal communication

Top correlated field pairs were identified as required by the submission template.

### 🟠 Workplace Stigma Index

Captures perceived negative consequences:

* Fear of employer reaction
* Observed discrimination
* Client impact
* Disclosure hesitation

Top 5 correlation pairs provided.

### 🔵 Organizational Openness Score

Captures comfort discussing mental health:

* With coworkers
* With supervisors
* With family/friends

Top 5 correlation pairs provided. All engineered features were saved into a modeling-ready dataset before clustering and supervised learning.

---

## 🤖 Supervised Modeling

Two required targets. Random Forest was evaluated but Logistic Regression performed better and provided clearer interpretability.

### 🎯 Model 1

Target:
**“Do you currently have a mental health disorder?”**

* Logistic Regression ROC-AUC: **0.923**
* F1 Score ≈ 0.90
* Top 10 correlated features identified and ranked
* Model trained only on selected top features 

Key predictors:

* Past disorder history
* Clinical diagnosis
* Family history
* Reported productivity interference

---

### 🎯 Model 2

Target:
**“Have you ever sought treatment for a mental health issue from a mental health professional?”**

* Logistic Regression ROC-AUC: **0.922**
* F1 Score ≈ 0.85
* Top 10 correlated features identified
* Model trained using only those selected predictors

Key drivers:

* Clinical diagnosis
* Awareness of available support resources
* Organizational openness
* Perceived stigma

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

Diagnosis-related variables showed near-perfect collinearity (r = 0.993 between diagnosis indicators).

---

## 🔍 Model Explainability (SHAP)

To ensure interpretability and strategic insight, SHAP (SHapley Additive exPlanations) was used to analyze feature contributions in both classification models.

### 💬 Treatment Model – Key Drivers

Treatment-seeking behavior is primarily influenced by:

* Employer mental health support
* Knowledge of available care options
* Perceived stigma and workplace interference

This provides actionable insights for organizational intervention strategies.

---

## 👥 Clustering 

Objective: Identify three distinct employee profiles. This employee segmentation used KMeans, k=3

1. **Clinically Diagnosed**
   High formal diagnosis rates and ongoing work interference.

2. **Undiagnosed but Impacted**
   Significant productivity loss without formal diagnosis, a hidden operational risk group.

3. **Lower Disclosure**
   Low reported openness and potential underreporting risk.


KMeans was selected because:

* Higher silhouette score than Agglomerative
* Stability on standardized numeric features
* Clear interpretability
* Suitable for fixed k=3 requirement

---

### Employee Segmentation – Behavioral Profiles

Using KMeans clustering (k=3), the workforce was segmented into three meaningful mental health profiles:

🔹 **Clinically Diagnosed & High Impact**
Employees with established diagnoses and significant workplace interference.

🔹 **Undiagnosed but Impacted**
Employees reporting productivity disruption without formal diagnosis, representing a hidden operational risk.

🔹 **Lower Risk / Moderate Support**
Employees with lower reported clinical burden and comparatively stronger perceived support.

This segmentation demonstrates that workplace mental health risk is not binary but distributed across distinct structural patterns. 

The identification of an undiagnosed yet operationally affected group underscores the importance of proactive screening and improved visibility of mental health resources.

Rather than applying uniform policies, organizations can tailor interventions based on employee risk profiles.

---

## 🤖 Model Serialization

Two predictive targets were modeled using Logistic Regression and Random Forest:

1. **Current mental health condition**
2. **Treatment-seeking behavior**

The trained models and their classification thresholds were serialized in the `models/` folder, enabling reproducible predictions without retraining. Serialized files include:

```
logistic_model_condition.pkl       logistic_threshold_condition.pkl
logistic_model_treatment.pkl       logistic_threshold_treatment.pkl
rf_model_condition.pkl             rf_threshold_condition.pkl
rf_model_treatment.pkl             rf_threshold_treatment.pkl
```

These can be loaded in Python using `joblib`:

```python
import joblib

# Load a model and its threshold
model = joblib.load('models/logistic_model_condition.pkl')
threshold = joblib.load('models/logistic_threshold_condition.pkl')
```

This approach ensures that future steps, analyses, or deployment pipelines can leverage the trained models directly, maintaining reproducibility and efficiency.

---

## 📈 Key Business Insights & Strategic Implications

The analysis reveals that mental health outcomes in technology workplaces are structurally driven rather than random.

Diagnosis history emerges as the strongest predictor of both current condition and treatment-seeking behavior, indicating continuity in mental health patterns. While perceived organizational support and openness culture influence outcomes, formal policy presence alone does not guarantee psychological safety. Similarly, company size does not inherently determine the quality of mental health support structures.

A particularly critical finding is the existence of a large segment of employees reporting measurable productivity impact without formal diagnosis. This group represents a hidden operational risk and a missed opportunity for early intervention.

These results suggest that organizations must move beyond reactive models of support and adopt proactive, data-driven strategies. Effective actions include:

* Implementing confidential early-risk screening mechanisms
* Increasing visibility and structured communication of available mental health resources
* Training managers to foster psychological safety and open dialogue
* Deploying targeted interventions tailored to distinct employee segments

By addressing both structural risk drivers and cultural dynamics, companies can reduce untreated cases, mitigate productivity loss, and strengthen long-term workforce sustainability.

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

Workplace mental health is predictable, measurable, and influenceable. The strong predictive signal (ROC-AUC > 0.92) combined with actionable segmentation demonstrates that data-driven strategies can meaningfully reduce untreated cases and productivity loss.

Mental health strategy is not only an ethical responsibility, it is a structural and economic lever for sustainable organizational performance.

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

#### 1️. Cleaning and analyze

```bash
python src/cleaning.py
python src/analyze_values.py
```

#### 2️. Feature Engineering

```bash
python src/features.py
```

#### 3. Modeling

```bash
python src/modeling_condition.py
python src/modeling_treatment.py
```
#### 4. Clustering

```bash
python src/clustering.py
```
#### 5. Submission of excel

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
* Final excel document 

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








