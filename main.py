import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')
print(df.head())

df.hist(figsize=(10,8), bins=20)
plt.suptitle("Distributions of Numeric Features", fontsize=14)
plt.show()