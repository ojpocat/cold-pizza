import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')

df.hist(figsize=(12,10), bins=20)
plt.suptitle("Distribution of All Numeric Features", fontsize=16)
# plt.show()

# Function to classify happiness levels
def categorize_happiness(h):
    if 4 <= h < 6:
        return "Low"
    elif 6 <= h < 8:
        return "Medium"
    elif 8 <= h <= 10:
        return "High"
    else:
        return None  # for values outside expected range

# Create the new column
df["Happiness_Group"] = df["Happiness_Index(1-10)"].apply(categorize_happiness)

scaler = StandardScaler()
df["Happiness_Standardized"] = scaler.fit_transform(
    df[["Happiness_Index(1-10)"]]
)

print(df)

# ============================================================
# CORRELATION BETWEEN SCREEN TIME & HAPPINESS
# ============================================================

corr = df["Daily_Screen_Time(hrs)"].corr(df["Happiness_Index(1-10)"])
print("Correlation between Screen Time and Happiness:", corr)
# negative correlation means more screen time -> less happiness

# ============================================================
# SCATTERPLOT: SCREEN TIME VS HAPPINESS
# ============================================================

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

# ============================================================
# REGRESSION LINE (LINEAR TREND)
# ============================================================

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

# more screen time generally reduces happiness

# ============================================================
# RELATIONSHIP WITH STANDARDIZED HAPPINESS
# ============================================================

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