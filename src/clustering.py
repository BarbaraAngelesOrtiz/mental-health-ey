# Mental Health Hackathon - Clustering
# Author: Barbara Ortiz
# Purpose:
#   - Identify statistically validated employee profiles
#   - Optimize number of clusters
#   - Provide interpretable cluster characterization

import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "mental_health_features.csv")
CLUSTER_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "analysis", "clusters.csv")
SUMMARY_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "analysis", "cluster_summary.csv")
METRICS_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "analysis", "cluster_metrics.txt")

# LOAD DATA

df = pd.read_csv(FEATURE_PATH)

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

df = df.rename(columns=rename_dict)

print("Loaded dataset shape:", df.shape)

# FEATURE SELECTION

features = [
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

features = [
    f for f in features
    if f in df.columns
    and pd.api.types.is_numeric_dtype(df[f])
    and df[f].notna().sum() > 0
]

if len(features) == 0:
    raise ValueError("No valid clustering features found after filtering.")

print("\nFeatures used for clustering:")
print(features)

X = df[features].copy()

X = X.apply(lambda col: col.fillna(col.median()))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# OPTIMAL K SELECTION (Silhouette Method)

silhouette_scores = {}
inertias = {}

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette_scores[k] = silhouette_score(X_scaled, labels)
    inertias[k] = kmeans.inertia_

best_k = max(silhouette_scores, key=silhouette_scores.get)

print(f"\nOptimal number of clusters (silhouette): {best_k}")

with open(METRICS_OUTPUT_PATH, "w") as f:
    f.write("Clustering Metrics\n\n")
    for k in silhouette_scores:
        f.write(f"K={k} | Silhouette={silhouette_scores[k]:.4f} | Inertia={inertias[k]:.2f}\n")
    f.write(f"\nBest K based on silhouette: {best_k}")

# FINAL CLUSTERING

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

df.to_csv(CLUSTER_OUTPUT_PATH, index=False)

# MODEL COMPARISON: KMeans vs Agglomerative

# KMeans
kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)

# Agglomerative
agg_model = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg_model.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, agg_labels)

print("\nModel Comparison:")
print(f"KMeans Silhouette Score:        {sil_kmeans:.4f}")
print(f"Agglomerative Silhouette Score: {sil_agg:.4f}")

# Select best model
if sil_kmeans >= sil_agg:
    print("\nSelected model: KMeans")
    df["cluster"] = kmeans_labels
    selected_model_name = "KMeans"
    selected_silhouette = sil_kmeans
else:
    print("\nSelected model: Agglomerative")
    df["cluster"] = agg_labels
    selected_model_name = "Agglomerative"
    selected_silhouette = sil_agg

# Save comparison metrics
with open(METRICS_OUTPUT_PATH, "a") as f:
    f.write("\n\nModel Comparison\n")
    f.write(f"KMeans Silhouette: {sil_kmeans:.4f}\n")
    f.write(f"Agglomerative Silhouette: {sil_agg:.4f}\n")
    f.write(f"Selected Model: {selected_model_name}\n")

# CLUSTER INTERPRETATION

cluster_means = df.groupby("cluster")[features].mean().round(3)
cluster_means.to_csv(SUMMARY_OUTPUT_PATH)

print("\nCluster means:")
print(cluster_means)

scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df["cluster"] = df["cluster"]

cluster_profiles = scaled_df.groupby("cluster").mean()

print("\nTOP 3 DEFINING FEATURES PER CLUSTER:")

for cluster_id in cluster_profiles.index:
    top3 = (
        cluster_profiles.loc[cluster_id]
        .sort_values(ascending=False)
        .head(3)
    )
    
    print(f"\nCluster {cluster_id}:")
    for feature, value in top3.items():
        print(f"  {feature}: {value:.2f} std from mean")

# PCA VISUALIZATION

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster"], alpha=0.6)
plt.title("Employee Profiles – PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "PCA.png"), bbox_inches="tight")
plt.show()

print("\n✅ Clustering completed successfully.")
print("Optimal K:", best_k)
print("Metrics saved to:", METRICS_OUTPUT_PATH)

# HEATMAP - STANDARDIZED CLUSTER PROFILES

# Standardize cluster means relative to global mean
global_means = df[features].mean()
global_std = df[features].std()

cluster_means_z = (cluster_means - global_means) / global_std

plt.figure(figsize=(10,5))
plt.imshow(cluster_means_z, aspect="auto")
plt.colorbar(label="Std from Global Mean")

plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(cluster_means_z.index)), cluster_means_z.index)

plt.title("Cluster Profiles (Standardized vs Global Mean)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "Cluster_profiles_zscore.png"))
plt.show()

# FEATURE CORRELATION HEATMAP

corr_matrix = df[features].corr()

plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "Feature_correlation_matrix.png"))
plt.show()

# RADAR PLOT PER CLUSTER

import numpy as np

labels = features
num_vars = len(labels)

angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for cluster_id in cluster_means.index:
    values = cluster_means.loc[cluster_id].tolist()
    values += values[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_rotation(np.degrees(angle))
        label.set_horizontalalignment("center")

    plt.title(f"Cluster {cluster_id} Profile Radar")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"images/Radar_cluster_{cluster_id}.png"))
    plt.show()

# CLUSTER DISTRIBUTION

cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Employees")
plt.title("Cluster Size Distribution")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images/Cluster_distribution.png"))
plt.show()
