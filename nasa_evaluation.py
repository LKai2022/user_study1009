import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------- 1. Data Preparation (Between-Subjects Design: 66 Participants) ----------------------
# 66 distinct participant IDs for a between-subjects study
participants = [f'P{i}' for i in range(1, 67)]
groups = ['Group A', 'Group B', 'Group C']

group_a_workload = [64.16666667, 61.33333333, 64.66666667, 62.5, 66.33333333, 64.16666667, 66.33333333, 64.33333333, 65.33333333, 63.5, 62.83333333, 59.33333333, 65.33333333, 59.66666667, 66.33333333, 66.33333333, 58.33333333, 61.33333333, 62.16666667, 58.33333333, 64.33333333, 66.66666667]
group_b_workload = [47.5, 49.33333333, 53, 51.33333333, 49.16666667, 53, 44.33333333, 51.33333333, 48, 47.5, 47.5, 51.5, 52.16666667, 44.66666667, 50.66666667, 53.66666667, 46.33333333, 43.66666667, 44.66666667, 48.16666667, 46, 46.33333333]
group_c_workload = [34.5, 36.3, 42, 39.3, 38.3, 40, 35.3, 36.3, 42.5, 39.7, 40, 40.8, 39.2, 42, 36.3, 43.2, 42.7, 38.5, 40.2, 37.7, 35.5, 37]

# Construct DataFrame
data = []
for i in range(22):
  data.append(['Group A', participants[i], group_a_workload[i]])
for i in range(22):
  data.append(['Group B', participants[22 + i], group_b_workload[i]])
for i in range(22):
  data.append(['Group C', participants[44 + i], group_c_workload[i]])

df = pd.DataFrame(data, columns=['Group', 'Participant', 'Workload_Score'])

# ---------------------- 2. Statistical Analysis ----------------------
group_data = {g: df[df['Group'] == g]['Workload_Score'] for g in groups}
group_list = [
    group_data['Group A'],
    group_data['Group B'],
    group_data['Group C'],
]

print('=' * 20 + ' 1. Assumption Checks ' + '=' * 20)

# 2.1 Normality Test (Shapiro-Wilk)
print('[Normality Test (Shapiro-Wilk)]:')
for g in groups:
  stat, p = stats.shapiro(group_data[g])
  normality_status = 'Passed' if p > 0.05 else 'Violated'
  print(f'  {g}: W = {stat:.4f}, p = {p:.4f} ({normality_status})')

# 2.2 Homogeneity of Variance Test (Levene)
stat_levene, p_levene = stats.levene(*group_list)
homogeneity_status = 'Passed' if p_levene > 0.05 else 'Violated'
print(
    f'\n[Homogeneity of Variance Test (Levene)]:\n  Statistic ='
    f' {stat_levene:.4f}, p = {p_levene:.4f} ({homogeneity_status})'
)

print('\n' + '=' * 20 + ' 2. One-Way ANOVA and Effect Size ' + '=' * 20)

# 2.3 One-Way ANOVA
f_stat, p_anova = stats.f_oneway(*group_list)
p_anova_str = '< 0.001' if p_anova < 0.001 else f'{p_anova:.4f}'
print(
    f'[One-Way ANOVA]:\n  F(2, 63) = {f_stat:.2f}\n  p-value  ='
    f' {p_anova_str}'
)

# 2.4 Effect Size Calculation (Eta-squared η²)
grand_mean = df['Workload_Score'].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_list)
ss_total = sum((df['Workload_Score'] - grand_mean) ** 2)
eta_squared = ss_between / ss_total
print(
    f'  Effect Size (Eta-squared η²) = {eta_squared:.4f}'
)

print('\n' + '=' * 20 + ' 3. Post-Hoc Pairwise Comparisons (Tukey HSD) ' + '=' * 20)

# 2.5 Tukey HSD Post-hoc Test
tukey = pairwise_tukeyhsd(
    endog=df['Workload_Score'], groups=df['Group'], alpha=0.05
)
tukey_summary = tukey.summary().data

print(
    f"{'Group 1':<10} {'Group 2':<10} {'Mean Diff':<12} {'p-adj':<12}"
    f" {'Reject Null':<12}"
)
print('-' * 60)

for row in tukey_summary[1:]:
  g1, g2, meandiff, p_adj, lower, upper, reject = row
  p_str = '< 0.001' if p_adj < 0.001 else f'{p_adj:.3f}'
  print(f'{g1:<10} {g2:<10} {meandiff:<12.4f} {p_str:<12} {str(reject):<12}')