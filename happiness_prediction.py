# ============================================================
# MACHINE LEARNING — Mental Health & Social Media
# Predicting Happiness & Identifying Lifestyle Clusters
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Clustering
from sklearn.cluster import KMeans

sns.set(style="whitegrid")

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
print("Data loaded. Shape:", df.shape)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# --- One-hot encode categorical features ---
gender_ohe = pd.get_dummies(df["Gender"], prefix="Gender")
platform_ohe = pd.get_dummies(df["Social_Media_Platform"], prefix="Platform")

df = pd.concat([df, gender_ohe, platform_ohe], axis=1)

# --- Add engineered features ---
df["ScreenSleep_Ratio"] = df["Daily_Screen_Time(hrs)"] / df["Sleep_Quality(1-10)"]
df["Stress_Exercise_Imbalance"] = df["Stress_Level(1-10)"] / (df["Exercise_Frequency(week)"] + 1)
df["Sleep_Stress_Interaction"] = df["Sleep_Quality(1-10)"] * df["Stress_Level(1-10)"]
df["Digital_Load"] = df["Daily_Screen_Time(hrs)"] + (10 - df["Days_Without_Social_Media"])

# --- Age grouping (categorical) ---
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 20, 30, 40, 55, 100],
    labels=["Teen", "20s", "30s", "40-55", "55+"]
)
age_group_ohe = pd.get_dummies(df["Age_Group"], prefix="AgeGroup")
df = pd.concat([df, age_group_ohe], axis=1)

# --- Platform category ---
platform_category_map = {
    "Instagram": "Visual",
    "TikTok": "Visual",
    "YouTube": "LongForm",
    "X (Twitter)": "Text",
    "LinkedIn": "Professional",
    "Facebook": "Social"
}
df["Platform_Category"] = df["Social_Media_Platform"].map(platform_category_map)
platform_cat_ohe = pd.get_dummies(df["Platform_Category"], prefix="PlatCat")
df = pd.concat([df, platform_cat_ohe], axis=1)

# ============================================================
# 3. PREPROCESSING
# ============================================================

# Define features and target
feature_cols = [
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)",
    "Age",
    # engineered features
    "ScreenSleep_Ratio",
    "Stress_Exercise_Imbalance",
    "Sleep_Stress_Interaction",
    "Digital_Load"
] + list(gender_ohe.columns) + list(platform_ohe.columns) + list(age_group_ohe.columns) + list(platform_cat_ohe.columns)

X = df[feature_cols]
y = df["Happiness_Index(1-10)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numeric features only
numeric_cols = [
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)",
    "Age",
    "ScreenSleep_Ratio",
    "Stress_Exercise_Imbalance",
    "Sleep_Stress_Interaction",
    "Digital_Load"
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("Preprocessing complete.\n")

# ============================================================
# 4. BASELINE MODEL — Linear Regression
# ============================================================

print("=== BASELINE: Linear Regression ===")

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

baseline_mse = mean_squared_error(y_test, y_pred_lr)
baseline_r2 = r2_score(y_test, y_pred_lr)

print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Baseline R²:  {baseline_r2:.4f}\n")

# ============================================================
# 5. MODEL 2 — Random Forest (with GridSearch)
# ============================================================

print("=== RANDOM FOREST REGRESSOR ===")

rf = RandomForestRegressor(random_state=42)

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 4, 6]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

rf_grid.fit(X_train_scaled, y_train)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)

print("Best RF Params:", rf_grid.best_params_)
print(f"RF R²:  {r2_score(y_test, y_pred_rf):.4f}")
print(f"RF MSE: {mean_squared_error(y_test, y_pred_rf):.4f}\n")

# ============================================================
# 6. MODEL 3 — SVR
# ============================================================

print("=== SUPPORT VECTOR REGRESSION ===")

svr = SVR()

svr_params = {
    "kernel": ["rbf", "poly"],
    "C": [0.5, 1, 2, 5],
    "gamma": ["scale", "auto"]
}

svr_grid = GridSearchCV(
    svr,
    svr_params,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

svr_grid.fit(X_train_scaled, y_train)

best_svr = svr_grid.best_estimator_
y_pred_svr = best_svr.predict(X_test_scaled)

print("Best SVR Params:", svr_grid.best_params_)
print(f"SVR R²:  {r2_score(y_test, y_pred_svr):.4f}")
print(f"SVR MSE: {mean_squared_error(y_test, y_pred_svr):.4f}\n")

# ============================================================
# 7. CROSS-VALIDATION COMPARISON
# ============================================================

print("=== CROSS-VALIDATION SCORES ===")

for model, name in [(lr, "Linear Regression"), (best_rf, "Random Forest"), (best_svr, "SVR")]:
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    print(f"{name}: Mean CV R² = {scores.mean():.4f} (std={scores.std():.4f})")

print()

# ============================================================
# 8. CLUSTERING (extended with engineered features)
# ============================================================

print("=== K-MEANS CLUSTERING ON FULL FEATURE SET ===")

X_scaled_full = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["ML_Cluster"] = kmeans.fit_predict(X_scaled_full)

cluster_summary = df.groupby("ML_Cluster")[feature_cols + ["Happiness_Index(1-10)"]].mean()
print(cluster_summary)

# Save cluster summary
cluster_summary.to_csv("cluster_summary.csv")

print("\nClustering complete. Saved cluster summary.\n")

# ============================================================
# 9. VISUALIZATION
# ============================================================

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["Daily_Screen_Time(hrs)"],
    y=df["Happiness_Index(1-10)"],
    hue=df["ML_Cluster"],
    palette="tab10"
)
plt.title("Clusters Based on Lifestyle Features")
plt.savefig("cluster_scatter.png")
plt.close()
