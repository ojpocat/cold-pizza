# ============================================================
# MENTAL HEALTH & SOCIAL MEDIA EDA NOTEBOOK
# ============================================================

# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

# Create output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')

# ============================================================
# 1. INITIAL DATA INSPECTION
# ============================================================

print(df.head())
print("\nColumns:", df.columns)
print("\nDataset info:\n")
print(df.info())
print("\nDescriptive statistics:\n")
print(df.describe())

# ============================================================
# 2. HAPPINESS GROUPING & STANDARDIZATION
# ============================================================

def categorize_happiness(h):
    if 4 <= h < 8:
        return "Low"
    elif 8 <= h < 10:
        return "Medium"
    elif 10 <= h <= 10:
        return "High"
    else:
        return None

df["Happiness_Group"] = df["Happiness_Index(1-10)"].apply(categorize_happiness)
df["Happiness_Standardized"] = StandardScaler().fit_transform(df[["Happiness_Index(1-10)"]])

# Plot distributions
plt.figure(figsize=(12,10))
df.hist(figsize=(12,10), bins=20)
plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.savefig(os.path.join(output_dir, "numeric_distributions.png"))
plt.close()

# ============================================================
# 3. SCREEN TIME VS HAPPINESS
# ============================================================

corr = df["Daily_Screen_Time(hrs)"].corr(df["Happiness_Index(1-10)"])
print("Correlation between Screen Time and Happiness:", corr)

# Scatterplot with happiness groups
plt.figure(figsize=(8,6))
sns.scatterplot(
    x="Daily_Screen_Time(hrs)",
    y="Happiness_Index(1-10)",
    hue="Happiness_Group",
    data=df,
    palette="viridis"
)
plt.title("Screen Time vs Happiness Index")
plt.xlabel("Daily Screen Time (hrs)")
plt.ylabel("Happiness Index")
plt.savefig(os.path.join(output_dir, "screen_vs_happiness_scatter.png"))
plt.close()

# Regression line
plt.figure(figsize=(8,6))
sns.regplot(
    x="Daily_Screen_Time(hrs)",
    y="Happiness_Index(1-10)",
    data=df,
    scatter_kws={'alpha':0.5},
    line_kws={'color':'red'}
)
plt.title("Screen Time vs Happiness with Linear Trend")
plt.savefig(os.path.join(output_dir, "screen_vs_happiness_regression.png"))
plt.close()

# Standardized happiness
plt.figure(figsize=(8,6))
sns.regplot(
    x="Daily_Screen_Time(hrs)",
    y="Happiness_Standardized",
    data=df,
    scatter_kws={'alpha':0.5},
    line_kws={'color':'red'}
)
plt.title("Screen Time vs Standardized Happiness")
plt.xlabel("Daily Screen Time (hrs)")
plt.ylabel("Happiness (Standardized)")
plt.savefig(os.path.join(output_dir, "screen_vs_standardized_happiness.png"))
plt.close()

# ============================================================
# 4. CLUSTERING: SCREEN-TIME LIFESTYLE GROUPS
# ============================================================

cluster_features = [
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)"
]

X_scaled = StandardScaler().fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Cluster summary
cluster_summary = df.groupby("Cluster")[cluster_features + ["Happiness_Index(1-10)"]].mean()
cluster_summary["Count"] = df.groupby("Cluster").size()
print("\nCluster Summary:\n", cluster_summary)

# Assign descriptive labels
def assign_cluster_label(row):
    if row["Daily_Screen_Time(hrs)"] > df["Daily_Screen_Time(hrs)"].mean() and \
       row["Sleep_Quality(1-10)"] < df["Sleep_Quality(1-10)"].mean():
        return "High Screen / Low Sleep"
    elif row["Exercise_Frequency(week)"] > df["Exercise_Frequency(week)"].mean() and \
         row["Stress_Level(1-10)"] < df["Stress_Level(1-10)"].mean():
        return "Healthy Lifestyle"
    else:
        return "Moderate Lifestyle"

cluster_summary = cluster_summary.reset_index()
cluster_summary["Cluster_Label"] = cluster_summary.apply(assign_cluster_label, axis=1)
label_map = cluster_summary.set_index("Cluster")["Cluster_Label"].to_dict()
df["Cluster_Label"] = df["Cluster"].map(label_map)

# ============================================================
# 5. VISUALIZING CLUSTERS
# ============================================================

# Scatterplot: Screen Time vs Happiness
plt.figure(figsize=(8,6))
sns.scatterplot(
    x="Daily_Screen_Time(hrs)",
    y="Happiness_Index(1-10)",
    hue="Cluster_Label",
    data=df,
    palette="Set1",
    alpha=0.7
)
plt.title("Screen Time vs Happiness by Cluster")
plt.xlabel("Daily Screen Time (hrs)")
plt.ylabel("Happiness Index")
plt.legend(title="Cluster")
plt.savefig(os.path.join(output_dir, "screen_vs_happiness_by_cluster.png"))
plt.close()

# Boxplots
features_to_plot = cluster_features + ["Happiness_Index(1-10)"]
plt.figure(figsize=(18,12))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="Cluster_Label", y=feature, data=df, palette="Set2", hue=None)
    plt.xticks(rotation=30)
    plt.title(f"{feature} by Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_feature_boxplots.png"))
plt.close()

# Pairplot
sns.pairplot(
    df,
    vars=features_to_plot,
    hue="Cluster_Label",
    palette="Set2",
    diag_kind="kde",
    height=2.5
)
plt.suptitle("Pairwise Relationships by Cluster", y=1.02)
plt.savefig(os.path.join(output_dir, "cluster_pairplot.png"))
plt.close()

# Average happiness per cluster
avg_happiness = df.groupby("Cluster_Label")["Happiness_Index(1-10)"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=avg_happiness.index, y=avg_happiness.values, palette="Set3")
plt.ylabel("Average Happiness")
plt.title("Average Happiness by Cluster")
plt.xticks(rotation=30)
plt.savefig(os.path.join(output_dir, "avg_happiness_by_cluster.png"))
plt.close()

# ============================================================
# 6. PLATFORM-BASED SCREEN TIME ANALYSIS
# ============================================================

# Boxplot: screen time per platform
plt.figure(figsize=(10,6))
sns.boxplot(x="Social_Media_Platform", y="Daily_Screen_Time(hrs)", data=df, palette="Set3", hue=None)
plt.xticks(rotation=45)
plt.title("Screen Time Distribution by Platform")
plt.savefig(os.path.join(output_dir, "screen_time_by_platform.png"))
plt.close()

# Average happiness per platform
platform_means = df.groupby("Social_Media_Platform")["Happiness_Index(1-10)"].mean()
plt.figure(figsize=(8,5))
sns.barplot(x=platform_means.index, y=platform_means.values, palette="Set3")
plt.ylabel("Average Happiness")
plt.title("Average Happiness by Platform")
plt.xticks(rotation=30)
plt.savefig(os.path.join(output_dir, "avg_happiness_by_platform.png"))
plt.close()

# ============================================================
# 7. AUTOMATIC CORRELATION LOOP
# ============================================================

def correlate_with_happiness(df, target_col="Happiness_Index(1-10)", strong_threshold=0.5):
    # Exclude unwanted columns
    exclude_cols = ["Cluster", "Happiness_Standardized", target_col]
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

    # Compute correlations
    correlations = {col: df[col].corr(df[target_col]) for col in numeric_cols}
    corr_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Correlation_with_Happiness"])
    corr_df["AbsCorrelation"] = corr_df["Correlation_with_Happiness"].abs()
    corr_df["Strong"] = corr_df["AbsCorrelation"] > strong_threshold
    corr_df = corr_df.sort_values(by="AbsCorrelation", ascending=False).reset_index(drop=True)

    # Positive correlations
    pos_corr = corr_df[corr_df["Correlation_with_Happiness"] > 0]
    if not pos_corr.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(x="Correlation_with_Happiness", y="Feature", data=pos_corr, palette="Greens_r")
        plt.title("Positive Correlations with Happiness")
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.subplots_adjust(left=0.29)
        plt.savefig(os.path.join(output_dir, "positive_correlations.png"))
        plt.close()

    # Negative correlations
    neg_corr = corr_df[corr_df["Correlation_with_Happiness"] < 0]
    if not neg_corr.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(x="Correlation_with_Happiness", y="Feature", data=neg_corr, palette="Reds_r")
        plt.title("Negative Correlations with Happiness")
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)  # increase left margin
        plt.savefig(os.path.join(output_dir, "negative_correlations.png"))
        plt.close()

    # Print strong correlations
    strong_corrs = corr_df[corr_df["Strong"]]
    print(f"Strong correlations (|corr| > {strong_threshold}):\n")
    print(strong_corrs[["Feature", "Correlation_with_Happiness"]])

    return corr_df

# Run function
corr_results = correlate_with_happiness(df)

