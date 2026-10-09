import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------- 1. Data Preparation (Between-Subjects Design: 66 Unique Participants) ----------------------
# 66 distinct participant IDs (22 participants per group)
participants = [f'P{i}' for i in range(1, 67)]
groups = ['Group A', 'Group B', 'Group C']

# SUS Scores (0-100 scale, higher indicates better usability)
group_a_sus = [45,42.5,47.5,47.5,35,50,40,50,52.5,35,35,37.5,40,42.5,47.5,52.5,37.5,40,35,47.5,55,45]
group_b_sus = [65,67.5,55,57.5,57.5,67.5,60,62.5,55,57.5,55,57.5,60,55,57.5,72.5,62.5,70,62.5,67.5,72.5,57.5]
group_c_sus = [82.5, 77.5, 80, 82.5, 80, 77.5, 82.5, 82.5, 75, 77.5, 85, 82.5, 80, 82.5, 75, 80, 75, 82.5, 77.5, 87.5, 85, 85]

# Construct Long-format DataFrame
data = []
for i in range(22):
  data.append(['Group A', participants[i], group_a_sus[i]])
for i in range(22):
  data.append(['Group B', participants[22 + i], group_b_sus[i]])
for i in range(22):
  data.append(['Group C', participants[44 + i], group_c_sus[i]])

df = pd.DataFrame(data, columns=['Group', 'Participant', 'SUS_Score'])

# ---------------------- 2. Statistical Analysis ----------------------
group_data = {g: df[df['Group'] == g]['SUS_Score'].values for g in groups}
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

print('\n' + '=' * 20 + ' 2. Non-Parametric Analysis (Kruskal-Wallis) ' + '=' * 20)

# 2.3 Kruskal-Wallis Test (Recommended due to normality/homogeneity violations)
stat_kw, p_kw = stats.kruskal(*group_list)
p_kw_str = '< 0.001' if p_kw < 0.001 else f'{p_kw:.4f}'
print(f'[Kruskal-Wallis Test]:\n  H Statistic = {stat_kw:.4f}\n  p-value     = {p_kw_str}')

print('\n' + '=' * 20 + ' 3. Post-Hoc Pairwise Comparisons (Mann-Whitney U with Bonferroni) ' + '=' * 20)

# 2.4 Pairwise Mann-Whitney U Tests with Bonferroni Correction
pairs = [('Group A', 'Group B'), ('Group A', 'Group C'), ('Group B', 'Group C')]
num_tests = len(pairs)

print(
    f"{'Group 1':<10} {'Group 2':<10} {'U Stat':<10} {'p-val (Raw)':<14}"
    f" {'p-val (Adj)':<14} {'Significance':<12}"
)
print('-' * 72)

for g1, g2 in pairs:
  u_stat, p_val = stats.mannwhitneyu(
      group_data[g1], group_data[g2], alternative='two-sided'
  )
  p_adj = min(p_val * num_tests, 1.0)  # Bonferroni adjustment

  p_raw_str = '< 0.001' if p_val < 0.001 else f'{p_val:.4f}'
  p_adj_str = '< 0.001' if p_adj < 0.001 else f'{p_adj:.4f}'
  sig_tag = 'Significant' if p_adj < 0.05 else 'ns'

  print(
      f'{g1:<10} {g2:<10} {u_stat:<10.1f} {p_raw_str:<14} {p_adj_str:<14}'
      f' {sig_tag:<12}'
  )
