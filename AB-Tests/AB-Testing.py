import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.signal.windows import tukey
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, 
                         mannwhitneyu, pearsonr, spearmanr, kendalltau, 
                         f_oneway, kruskal)
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Sampling
population = np.random.randint(0, 80, 10000)
np.random.seed(115)
sample = np.random.choice(a=population, size=100)
print("Sample Mean:", sample.mean())

# Descriptive Statistics

df = sns.load_dataset("tips")
print(df.describe().T)

# Confidence Interval
print(sms.DescrStatsW(df["total_bill"]).tconfint_mean())
print(sms.DescrStatsW(df["tip"]).tconfint_mean())

# Correlation Analysis
df["total_bill"] = df["total_bill"] - df["tip"]
df.plot.scatter("tip", "total_bill")
plt.show()
print("Correlation:", df["tip"].corr(df["total_bill"]))

# Hypothesis Testing
# Independent Two-Sample T-Test

df = sns.load_dataset("tips")

# 1. Define Hypotheses
# H0: M1 = M2 (No significant difference)
# H1: M1 != M2 (Significant difference exists)

# 2. Assumption Checks
# Normality Assumption
print("Normality Test for Smokers")
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

# Variance Homogeneity
print("Levene Test for Homogeneity of Variance")
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

# Applying Hypothesis Test
print("Independent T-Test")
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"], equal_var=True)
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

print("Mann-Whitney U Test")
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

# Two-Proportion Z-Test
print("Two-Proportion Z-Test")
success_counts = np.array([300, 250])
observations = np.array([1000, 1100])

z_stat, pvalue = proportions_ztest(count=success_counts, nobs=observations)
print(f"Z-Stat: {z_stat:.4f}, p-value: {pvalue:.4f}")

# ANOVA: Comparing More Than Two Groups
print("ANOVA Test")
for group in df["day"].unique():
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(f"{group}: p-value {pvalue:.4f}")

# Variance Homogeneity
print("Levene Test for ANOVA")
test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print(f"Test Stat: {test_stat:.4f}, p-value: {pvalue:.4f}")

# Parametric ANOVA Test
print("One-Way ANOVA")
f_stat, pvalue = f_oneway(df.loc[df["day"] == "Sun", "total_bill"],
                          df.loc[df["day"] == "Sat", "total_bill"],
                          df.loc[df["day"] == "Thur", "total_bill"],
                          df.loc[df["day"] == "Fri", "total_bill"])
print(f"F-Stat: {f_stat:.4f}, p-value: {pvalue:.4f}")

# Non-Parametric Kruskal-Wallis Test
print("Kruskal-Wallis Test")
kruskal_stat, pvalue = kruskal(df.loc[df["day"] == "Sun", "total_bill"],
                               df.loc[df["day"] == "Sat", "total_bill"],
                               df.loc[df["day"] == "Thur", "total_bill"],
                               df.loc[df["day"] == "Fri", "total_bill"])
print(f"Test Stat: {kruskal_stat:.4f}, p-value: {pvalue:.4f}")

# Post-Hoc Analysis with Tukey Test
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
