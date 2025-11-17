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
