import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Pandas display options for better visualization
# Ensure all columns are shown when displaying a DataFrame
pd.set_option('display.max_columns', None)
# Adjust the display width to avoid truncation
pd.set_option('display.width', 500)

# Load Titanic dataset using Seaborn's built-in dataset loader
df = sns.load_dataset("titanic")

# Plot the count of passengers by gender
df["sex"].value_counts().plot(kind="bar")
plt.title("Count of Passengers by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Plot the age distribution of passengers
plt.hist(df["age"], bins=20, color="blue", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot to visualize the distribution of fare prices
plt.boxplot(df["fare"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

# Create and plot two different data sets to demonstrate line plots
y = np.array([13, 28, 11, 100])
x = np.array([21, 45, 3, 250])

# First subplot: Plot dataset 1
plt.subplot(1, 2, 1)
plt.title("Plot 1")
plt.plot(x, y, linestyle="dashed", color="r")
plt.xlabel("DataX")
plt.ylabel("DataY")
plt.grid()

# Second subplot: Plot dataset 2
y2 = np.array([1, 8, 111, 5])
x2 = np.array([12, 82, 33, 20])
plt.subplot(1, 2, 2)
plt.title("Plot 2")
plt.plot(x2, y2, linestyle="dashed", color="g")
plt.xlabel("DataX")
plt.ylabel("DataY")
plt.grid()

# Adjust layout to avoid overlapping and display the plots
plt.tight_layout()
plt.show()

# Count plot using Seaborn to visualize passenger gender distribution
sns.countplot(x=df["sex"], data=df)
plt.title("Count of Passengers by Gender (Seaborn)")
plt.show()

# Boxplot using Seaborn to visualize fare distribution
sns.boxplot(x=df["fare"])
plt.title("Fare Distribution (Seaborn)")
plt.show()
