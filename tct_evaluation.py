import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------- 1. Data Preparation ----------------------
# Define unique participant IDs for independent samples (22 per group, 66 total)
participants_a = [f'A_{i:02d}' for i in range(1, 23)]
participants_b = [f'B_{i:02d}' for i in range(1, 23)]
participants_c = [f'C_{i:02d}' for i in range(1, 23)]

groups = ['Group A', 'Group B', 'Group C']

group_a_tct = [290.6, 283.5, 271.8, 269.2, 278.8, 301.2, 290.6, 276.9, 269.7, 280.8, 293.2, 290.6, 283.7, 290.1, 276.6, 270.7, 274.8, 280.1, 287.8, 300.4, 293.5, 277.9]
group_b_tct = [250.2, 251.2, 270.5, 268.6, 252.6, 265.4, 263.5, 260.8, 250.6, 248.6, 263.3, 254.7, 263.5, 261.8, 253.6, 282.4, 278.4, 271.5, 255.6, 258.3, 269.4, 265.9]
group_c_tct = [237.7, 231.8, 263.2, 252.5, 232.9, 255.5, 243.2, 255.4, 251.1, 251.8, 256.2, 243.5, 245.7, 246.5, 244.6, 250.7, 245.6, 249.7, 244.6, 254.5, 237.1, 241.9]

# Construct DataFrame
data = []
for i in range(22):
    data.append(['Group A', participants_a[i], group_a_tct[i]])
    data.append(['Group B', participants_b[i], group_b_tct[i]])
    data.append(['Group C', participants_c[i], group_c_tct[i]])

df = pd.DataFrame(data, columns=['Group', 'Participant', 'Completion Time'])

# ---------------------- 2. Descriptive Statistics ----------------------
print("=" * 20 + " 1. Descriptive Statistics " + "=" * 20)
for g in groups:
    g_time = df[df['Group'] == g]['Completion Time']
    print(f"{g} -> Mean: {g_time.mean():.3f}, Variance: {g_time.var():.3f}, SD: {g_time.std():.3f}")

# ---------------------- 3. Assumption Tests ----------------------
group_data = {g: df[df['Group'] == g]['Completion Time'] for g in groups}
group_list = [group_data['Group A'], group_data['Group B'], group_data['Group C']]

print("\n" + "=" * 20 + " 2. Assumption Testing " + "=" * 20)

# 3.1 Normality Test (Shapiro-Wilk)
print("[Normality Test (Shapiro-Wilk)]:")
for g in groups:
    stat, p = stats.shapiro(group_data[g])
    status = "(Normal)" if p > 0.05 else "(Non-Normal)"
    print(f"  {g}: W = {stat:.4f}, p = {p:.4f} {status}")

# 3.2 Homogeneity of Variance Test (Levene's Test)
stat_levene, p_levene = stats.levene(*group_list)
homogeneity_status = "(Equal Variances)" if p_levene > 0.05 else "(Unequal Variances)"
print(f"\n[Homogeneity of Variance (Levene)]:\n  Statistic = {stat_levene:.4f}, p = {p_levene:.4f} {homogeneity_status}")

# ---------------------- 4. One-Way ANOVA & Effect Size ----------------------
print("\n" + "=" * 20 + " 3. Analysis of Variance (ANOVA) " + "=" * 20)

# 4.1 One-Way Independent ANOVA
f_stat, p_anova = stats.f_oneway(*group_list)
df_between = len(groups) - 1
df_within = len(df) - len(groups)
print(f"[One-Way ANOVA]:\n  F({df_between}, {df_within}) = {f_stat:.2f}\n  p-value = {p_anova:.4e}")

# 4.2 Effect Size (Eta-squared, η²)
grand_mean = df['Completion Time'].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_list)
ss_total = sum((df['Completion Time'] - grand_mean)**2)
eta_squared = ss_between / ss_total
print(f"  Effect Size (Eta-squared η²) = {eta_squared:.4f} ")

# ---------------------- 5. Post-Hoc Pairwise Comparisons ----------------------
print("\n" + "=" * 20 + " 4. Post-Hoc Comparisons (Tukey HSD) " + "=" * 20)

tukey = pairwise_tukeyhsd(endog=df['Completion Time'], groups=df['Group'], alpha=0.05)
tukey_summary = tukey.summary().data

# Formatted Table Header
print(f"{'Group 1':<12} {'Group 2':<12} {'Mean Diff':<12} {'p-adj':<14} {'Reject H0':<10}")
print("-" * 60)

# Iterate through Tukey summary table
for row in tukey_summary[1:]:
    g1, g2, meandiff, p_adj, lower, upper, reject = row
    meandiff_val = float(meandiff)
    p_adj_val = float(p_adj)

    # Format p-value according to standard academic reporting
    p_str = "< 0.001" if p_adj_val < 0.001 else f"{p_adj_val:.3f}"

    print(f"{g1:<12} {g2:<12} {meandiff_val:<12.4f} {p_str:<14} {str(reject):<10}")