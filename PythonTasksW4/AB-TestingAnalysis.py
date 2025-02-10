import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.stats.api as sms
from scipy.signal.windows import tukey
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

from AB_Testing import pvalue

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Task 1: Data Preparation and Analysis
# Step 1: Read the dataset containing control and test group data

control_group_path = r"C:\Users\hhfid\Desktop\miuul\measurement_problems\ab_testing\ab_testing.xlsx"
test_group_path = r"C:\Users\hhfid\Desktop\miuul\measurement_problems\ab_testing\ab_testing.xlsx"

df_c = pd.read_excel(control_group_path, sheet_name="Control Group")
df_t = pd.read_excel(test_group_path, sheet_name="Test Group")

# Step 2: Analyze the control and test group data
df_t.describe().T

# Step 3: Merge the control and test group data

df_t.columns = ["Impression_T", "Click_T", "Purchase_T", "Earning_T"]
combined = pd.concat([df_c, df_t], axis=1)

# Task 2: Defining the A/B Test Hypothesis
# H0: There is no significant difference in purchase means between Maximum Bidding and Average Bidding.
# H1: There is a significant difference.

# Step 2: Analyze the mean purchase values for control and test groups

print("Control Group Purchase Mean:", combined["Purchase"].mean())
print("Test Group Purchase Mean:", combined["Purchase_T"].mean())

# Task 3: Performing Hypothesis Testing
# Step 1: Check Assumptions (Normality & Homogeneity of Variances)

# Normality Test (Shapiro-Wilk Test)
test_stat, pvalue = shapiro(combined["Purchase"])
print("Shapiro-Wilk Test (Control): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(combined["Purchase_T"])
print("Shapiro-Wilk Test (Test): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Homogeneity of Variance Test (Levene's Test)
test_stat, pvalue = levene(combined["Purchase"], combined["Purchase_T"])
print("Levene's Test: test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Step 2: Choose the appropriate test based on assumptions
# Since assumptions are met, we proceed with an independent t-test

test_stat, pvalue = ttest_ind(combined["Purchase"], combined["Purchase_T"], equal_var=True)
print("T-test: test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Step 3: Interpret the results
# If p-value > 0.05, we fail to reject H0, meaning there is no statistically significant difference.

# Task 4: Analyzing Additional Metrics
# Step 1: Investigate other potential insights (e.g., clicks and impressions)

test_stat, pvalue = shapiro(combined["Click"])
print("Shapiro-Wilk Test (Clicks - Control): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(combined["Click_T"])
print("Shapiro-Wilk Test (Clicks - Test): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Variance homogeneity for clicks
test_stat, pvalue = levene(combined["Click"], combined["Click_T"])
print("Levene's Test (Clicks): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# T-test for clicks
test_stat, pvalue = ttest_ind(combined["Click"], combined["Click_T"], equal_var=False)
print("T-test (Clicks): test-stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Recommendation: Since no significant difference is found in purchases, consider focusing on ad impressions or clicks for optimization.
