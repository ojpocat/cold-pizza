import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')

# ============================================================
# DISTRIBUTION OF NUMERIC FEATURES
# ============================================================

df.hist(figsize=(12,10), bins=20)
plt.suptitle("Distribution of All Numeric Features", fontsize=16)

# ============================================================
# HAPPINESS GROUPING & STANDARDIZATION
# ============================================================

# Function to classify happiness levels
def categorize_happiness(h):
    if 4 <= h < 6:
        return "Low"
    elif 6 <= h < 8:
        return "Medium"
    elif 8 <= h <= 10:
        return "High"
    else:
        return None

# Create the new column
df["Happiness_Group"] = df["Happiness_Index(1-10)"].apply(categorize_happiness)

scaler = StandardScaler()
df["Happiness_Standardized"] = scaler.fit_transform(
    df[["Happiness_Index(1-10)"]]
)

print(df)
print(df.columns)

# ============================================================
# SCREEN TIME VS HAPPINESS
# ============================================================

# Correlation
corr = df["Daily_Screen_Time(hrs)"].corr(df["Happiness_Index(1-10)"])
print("Correlation between Screen Time and Happiness:", corr)

# Scatterplot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x="Daily_Screen_Time(hrs)", 
    y="Happiness_Index(1-10)", 
    data=df,
    hue="Happiness_Group",
    palette="viridis"
)
plt.title("Screen Time vs Happiness Index")
plt.xlabel("Daily Screen Time (hrs)")
plt.ylabel("Happiness Index")
plt.show()

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
plt.show()

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
plt.show()

# Features for clustering
features = [
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)"
]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# KMeans clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ============================================================
# CLUSTERS
# ============================================================

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="Daily_Screen_Time(hrs)",
    y="Happiness_Index(1-10)",
    hue="Cluster",
    palette="Set1",
    data=df
)
plt.title("Clusters Based on Screen-Time Lifestyle Features")
plt.xlabel("Daily Screen Time (hrs)")
plt.ylabel("Happiness Index")
plt.show()

# EXAMINE CLUSTER MEANS

cluster_summary = df.groupby("Cluster")[features + ["Happiness_Index(1-10)"]].mean()
cluster_summary["Count"] = df.groupby("Cluster").size()
print("Cluster Summary:\n")
print(cluster_summary)

# INTERPRET CLUSTERS

# Function to assign descriptive labels based on observed means
def describe_cluster(row):
    if row["Daily_Screen_Time(hrs)"] > df["Daily_Screen_Time(hrs)"].mean() and \
       row["Sleep_Quality(1-10)"] < df["Sleep_Quality(1-10)"].mean():
        return "High Screen Time / Poor Sleep"
    elif row["Exercise_Frequency(week)"] > df["Exercise_Frequency(week)"].mean() and \
         row["Stress_Level(1-10)"] < df["Stress_Level(1-10)"].mean():
        return "Healthy Lifestyle"
    else:
        return "Moderate Screen Time / Mixed Wellness"

cluster_summary["Description"] = cluster_summary.apply(describe_cluster, axis=1)
print("\nCluster Descriptions:\n")
print(cluster_summary[["Description", "Count"]])

# ADD CLUSTER DESCRIPTION TO DATAFRAME

# Map cluster numbers to descriptive labels
cluster_map = cluster_summary["Description"].to_dict()
df["Cluster_Description"] = df["Cluster"].map(cluster_map)

# Quick check
df[["User_ID", "Cluster", "Cluster_Description", "Happiness_Index(1-10)", "Daily_Screen_Time(hrs)"]].head()

# ============================================================
# PLATFORM-BASED SCREEN TIME ANALYSIS
# ============================================================

plt.figure(figsize=(10,6))
sns.boxplot(
    x="Social_Media_Platform",
    y="Daily_Screen_Time(hrs)",
    data=df,
    palette="Set3"
)
plt.xticks(rotation=45)
plt.title("Screen Time Distribution by Social Media Platform")
plt.show()

# Average happiness per platform
platform_means = df.groupby("Social_Media_Platform")["Happiness_Index(1-10)"].mean()
platform_means.plot(kind="bar", figsize=(8,5), color="skyblue")
plt.ylabel("Average Happiness")
plt.title("Average Happiness by Platform")
plt.show()