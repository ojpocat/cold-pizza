# ============================================================
# ML PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================

print("="*60)
print("OPTIMIZED ML PIPELINE")
print("="*60)

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
print(f"Data shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

df_enhanced = df.copy()

print("\nCreating engineered features...")

df_enhanced["Sleep_Stress_Ratio"] = df_enhanced["Sleep_Quality(1-10)"] / (
    df_enhanced["Stress_Level(1-10)"] + 0.1
)
df_enhanced["ScreenSleep_Ratio"] = df_enhanced["Daily_Screen_Time(hrs)"] / (
    df_enhanced["Sleep_Quality(1-10)"] + 0.1
)
df_enhanced["Wellness_Score"] = (
    df_enhanced["Sleep_Quality(1-10)"] * 0.4 +
    df_enhanced["Exercise_Frequency(week)"] * 0.3 +
    (10 - df_enhanced["Stress_Level(1-10)"]) * 0.3
)
df_enhanced["Recovery_Potential"] = (
    (df_enhanced["Sleep_Quality(1-10)"] + df_enhanced["Exercise_Frequency(week)"]) /
    (df_enhanced["Stress_Level(1-10)"] + 0.1)
)
df_enhanced["Age_ScreenTime_Interaction"] = (
    df_enhanced["Age"] * df_enhanced["Daily_Screen_Time(hrs)"]
)

# Original + engineered features
original_features = [
    "Daily_Screen_Time(hrs)", "Sleep_Quality(1-10)", "Stress_Level(1-10)",
    "Days_Without_Social_Media", "Exercise_Frequency(week)", "Age"
]
engineered_features = [
    "Sleep_Stress_Ratio", "ScreenSleep_Ratio",
    "Wellness_Score", "Recovery_Potential", "Age_ScreenTime_Interaction"
]
all_features = original_features + engineered_features
print(f"Using {len(all_features)} features")

X = df_enhanced[all_features].copy()
y = df_enhanced["Happiness_Index(1-10)"].copy()

# ============================================================
# 3. PREPROCESSING
# ============================================================

print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. MODEL TRAINING w/ OPTIMAL PARAMETERS
# ============================================================

print("\n" + "="*60)
print("MODEL TRAINING & OPTIMIZATION")
print("="*60)

models = {}
results = []

# -----------------------------
# 1. Gradient Boosting
# -----------------------------
print("\n[1] Gradient Boosting")
gbr = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3,
    max_features='sqrt',
    min_samples_split=10,
    n_estimators=300,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gbr
y_pred_gb = gbr.predict(X_test_scaled)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
results.append(('Gradient Boosting', gb_r2, gb_mae))
print(f"R²:  {gb_r2:.4f} | MAE: {gb_mae:.4f}")

# -----------------------------
# 2. Random Forest
# -----------------------------
print("\n[2] Random Forest")
rf = RandomForestRegressor(
    bootstrap=True,
    max_depth=6,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf
y_pred_rf = rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
results.append(('Random Forest', rf_r2, rf_mae))
print(f"R²:  {rf_r2:.4f} | MAE: {rf_mae:.4f}")

# -----------------------------
# 3. Support Vector Regression
# -----------------------------
print("\n[3] Support Vector Regression")
svr = SVR(
    kernel='rbf',
    C=10,
    degree=2,
    epsilon=0.1,
    gamma=0.01
)
svr.fit(X_train_scaled, y_train)
models['SVR'] = svr
y_pred_svr = svr.predict(X_test_scaled)
svr_r2 = r2_score(y_test, y_pred_svr)
svr_mae = mean_absolute_error(y_test, y_pred_svr)
results.append(('Support Vector Regression', svr_r2, svr_mae))
print(f"R²:  {svr_r2:.4f} | MAE: {svr_mae:.4f}")

# -----------------------------
# 4. Kernel Ridge Regression
# -----------------------------
print("\n[4] Kernel Ridge Regression")
krr = KernelRidge(
    kernel='polynomial'
)
krr.fit(X_train_scaled, y_train)
models['Kernel Ridge'] = krr
y_pred_krr = krr.predict(X_test_scaled)
krr_r2 = r2_score(y_test, y_pred_krr)
krr_mae = mean_absolute_error(y_test, y_pred_krr)
results.append(('Kernel Ridge', krr_r2, krr_mae))
print(f"R²:  {krr_r2:.4f} | MAE: {krr_mae:.4f}")

# ============================================================
# 5. RESULTS ANALYSIS
# ============================================================

results_df = pd.DataFrame(results, columns=['Model', 'R² Score', 'MAE'])
results_df = results_df.sort_values('R² Score', ascending=False)
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print("\n" + results_df.to_string(index=False))

# Feature importance for Gradient Boosting
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': models['Gradient Boosting'].feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("TOP FEATURE IMPORTANCES")
print("="*60)
print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:30} {row['Importance']:.4f}")

# ============================================================
# 6. PREDICTION ANALYSIS
# ============================================================

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

pred_actual_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best,
    'Error': y_test.values - y_pred_best
})

print(f"\nBest Model: {best_model_name}")
print("\nSample Predictions (first 10 test samples):")
for i in range(min(10, len(pred_actual_df))):
    print(f"  Actual: {pred_actual_df['Actual'].iloc[i]:.1f} | "
          f"Predicted: {pred_actual_df['Predicted'].iloc[i]:.1f} | "
          f"Error: {pred_actual_df['Error'].iloc[i]:+.2f}")

mean_error = pred_actual_df['Error'].mean()
std_error = pred_actual_df['Error'].std()
within_1_point = (abs(pred_actual_df['Error']) <= 1.0).sum() / len(pred_actual_df) * 100

print(f"\nError Statistics:")
print(f"  Mean Error: {mean_error:+.3f}")
print(f"  Std Error: {std_error:.3f}")
print(f"  Predictions within ±1 point: {within_1_point:.1f}%")

# ============================================================
# 7. SAVE VISUALIZATION
# ============================================================

# Plot prediction errors for best model
plt.figure(figsize=(10, 5))
sns.histplot(pred_actual_df['Error'], bins=20, kde=True, color='#F18F01')
plt.title(f'Prediction Error Distribution — {best_model_name}', fontweight='bold')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean Error = {mean_error:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_error_distribution.png', dpi=120)
plt.show()

# Plot R² and MAE for all models
fig, ax1 = plt.subplots(figsize=(10, 5))

color_r2 = '#2E86AB'
color_mae = '#A23B72'

ax1.bar(results_df['Model'], results_df['R² Score'], color=color_r2, alpha=0.7, label='R² Score')
ax1.set_ylabel('R² Score', color=color_r2, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='y', labelcolor=color_r2)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(results_df['Model'], results_df['MAE'], color=color_mae, marker='o', linewidth=2, label='MAE')
ax2.set_ylabel('Mean Absolute Error', color=color_mae, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_mae)

plt.title('Model Performance: R² vs MAE', fontweight='bold')
fig.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=120)
plt.show()

# ============================================================
# 8. HAPPINESS CLASSIFICATION
# ============================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge  # For regression only; skip
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "="*60)
print("HAPPINESS CLASSIFICATION: Low / Medium / High")
print("="*60)

# Bin happiness into 3 categories (Quantile-based bins)
df_enhanced['Happiness_Bin'] = pd.qcut(
    df_enhanced['Happiness_Index(1-10)'],
    q=3,
    labels=['Low', 'Medium', 'High']
)


y_class = df_enhanced['Happiness_Bin']

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Scale features (same scaler as regression)
X_train_scaled_c = scaler.fit_transform(X_train_c)
X_test_scaled_c = scaler.transform(X_test_c)

# Dictionary to store classifiers and results
classifiers = {}
class_results = []

# -----------------------------
# 1. Gradient Boosting Classifier
# -----------------------------
gbc = GradientBoostingClassifier(
    learning_rate=0.01,
    max_depth=3,
    max_features='sqrt',
    min_samples_split=10,
    n_estimators=300,
    subsample=0.8,
    random_state=42
)
gbc.fit(X_train_scaled_c, y_train_c)
y_pred_gbc = gbc.predict(X_test_scaled_c)
acc_gbc = accuracy_score(y_test_c, y_pred_gbc)
class_results.append(('Gradient Boosting', acc_gbc))
classifiers['Gradient Boosting'] = gbc

# -----------------------------
# 2. Random Forest Classifier
# -----------------------------
rfc = RandomForestClassifier(
    bootstrap=True,
    max_depth=6,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rfc.fit(X_train_scaled_c, y_train_c)
y_pred_rfc = rfc.predict(X_test_scaled_c)
acc_rfc = accuracy_score(y_test_c, y_pred_rfc)
class_results.append(('Random Forest', acc_rfc))
classifiers['Random Forest'] = rfc

# -----------------------------
# 3. Support Vector Classifier
# -----------------------------
svc = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01
)
svc.fit(X_train_scaled_c, y_train_c)
y_pred_svc = svc.predict(X_test_scaled_c)
acc_svc = accuracy_score(y_test_c, y_pred_svc)
class_results.append(('SVC', acc_svc))
classifiers['SVC'] = svc

# -----------------------------
# 4. Results Summary
# -----------------------------
class_results_df = pd.DataFrame(class_results, columns=['Model', 'Accuracy'])
class_results_df = class_results_df.sort_values('Accuracy', ascending=False)
print("\nClassification Accuracy (Happiness Bins):")
print(class_results_df.to_string(index=False))

# Confusion matrix for best classifier
best_clf_name = class_results_df.iloc[0]['Model']
best_clf = classifiers[best_clf_name]
y_pred_best_clf = best_clf.predict(X_test_scaled_c)

print(f"\nBest Classifier: {best_clf_name}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_c, y_pred_best_clf))
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_best_clf))

# Plot classification accuracy
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=class_results_df, palette='muted')
plt.ylim(0, 1)
plt.title('Classifier Accuracy on Binned Happiness')
plt.ylabel('Accuracy')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('classification_accuracy.png', dpi=120)
plt.show()