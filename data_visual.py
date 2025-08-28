
# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset safely with error handling
try:
    iris = load_iris(as_frame=True)  # loads as pandas DataFrame
    df = iris.frame
except FileNotFoundError:
    print("Dataset not found. Please check the file path.")
except Exception as e:
    print("Error loading dataset:", e)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check structure, data types, and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (Iris has no missing values, but just in case)
df = df.dropna()

# ---------------------------------
# Task 2: Basic Data Analysis
# ---------------------------------

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
print("\nAverage values by species:")
print(df.groupby("target").mean())

# Map target numbers to species names for readability
df["species"] = df["target"].map(dict(zip(range(3), iris.target_names)))

# Example finding:
# - Versicolor has longer petal length than Setosa
# - Virginica tends to have the largest features overall

# ---------------------------------
# Task 3: Data Visualization
# ---------------------------------

# 1. Line Chart - Just to show trend across samples
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Sepal vs Petal Length over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(6,4))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="Set2")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - Distribution of sepal length
plt.figure(figsize=(6,4))
plt.hist(df["sepal length (cm)"], bins=15, edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal length vs Petal length
plt.figure(figsize=(6,4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
