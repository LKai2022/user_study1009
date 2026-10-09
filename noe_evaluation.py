import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ---------------------- 1. Data Preparation (Between-Subjects Design: 66 Participants) ----------------------
group_a_errors = [1, 2, 0, 3, 2, 0, 2, 1, 1, 3, 4, 2, 2, 2, 1, 4, 3, 3, 1, 2, 4, 2]
group_b_errors = [1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 1, 2, 1, 0, 2, 2, 0, 2, 1, 1, 1, 1]
group_c_errors = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]

# Create unique participant IDs for 66 distinct subjects
participants = [f'P{i}' for i in range(1, 67)]

data = []
for i in range(22):
  data.append(['Group A', participants[i], group_a_errors[i]])
for i in range(22):
  data.append(['Group B', participants[22 + i], group_b_errors[i]])
for i in range(22):
  data.append(['Group C', participants[44 + i], group_c_errors[i]])

df = pd.DataFrame(data, columns=['Group', 'Participant', 'Error_Count'])

# ---------------------- 2. Poisson GLM Analysis ----------------------
print('=' * 20 + ' 1. Poisson GLM Analysis ' + '=' * 20)

# 2.1 Fit the primary Poisson GLM model
poisson_model = smf.glm(
    'Error_Count ~ C(Group)', data=df, family=sm.families.Poisson()
).fit()

# Compute Global Likelihood Ratio Chi-Square Statistic
null_model = smf.glm(
    'Error_Count ~ 1', data=df, family=sm.families.Poisson()
).fit()
lr_stat = 2 * (poisson_model.llf - null_model.llf)

lr_p = stats.chi2.sf(lr_stat, df=2)
lr_p_str = '< 0.001' if lr_p < 0.001 else f'{lr_p:.3f}'

print('[Poisson GLM Global Likelihood Ratio Test]:')
print(f'  LR Chi2(2) = {lr_stat:.4f}')
print(f'  p-value    = {lr_p_str}')

# 2.2 Pairwise Post-hoc Comparisons (with Holm-Bonferroni Correction)
poisson_a = smf.glm(
    'Error_Count ~ C(Group, Treatment(reference="Group A"))',
    data=df,
    family=sm.families.Poisson(),
).fit()
poisson_b = smf.glm(
    'Error_Count ~ C(Group, Treatment(reference="Group B"))',
    data=df,
    family=sm.families.Poisson(),
).fit()

coeff_B_vs_A = poisson_a.params[
    'C(Group, Treatment(reference="Group A"))[T.Group B]'
]
coeff_C_vs_A = poisson_a.params[
    'C(Group, Treatment(reference="Group A"))[T.Group C]'
]
coeff_C_vs_B = poisson_b.params[
    'C(Group, Treatment(reference="Group B"))[T.Group C]'
]

p_A_vs_B = poisson_a.pvalues[
    'C(Group, Treatment(reference="Group A"))[T.Group B]'
]
p_A_vs_C = poisson_a.pvalues[
    'C(Group, Treatment(reference="Group A"))[T.Group C]'
]
p_B_vs_C = poisson_b.pvalues[
    'C(Group, Treatment(reference="Group B"))[T.Group C]'
]

raw_p = [p_A_vs_B, p_A_vs_C, p_B_vs_C]
rrs = [np.exp(coeff_B_vs_A), np.exp(coeff_C_vs_A), np.exp(coeff_C_vs_B)]

reject, adj_p, _, _ = multipletests(raw_p, method='holm')

pairs = [('Group A', 'Group B'), ('Group A', 'Group C'), ('Group B', 'Group C')]

print(
    '\n'
    + '=' * 20
    + ' 2. Post-hoc Pairwise Comparisons (Holm-Bonferroni Adjusted) '
    + '=' * 20
)
print(
    f"{'Group 1':<10} {'Group 2':<10} {'Rate Ratio':<12} {'Raw p':<10}"
    f" {'Adj p':<10} {'Sig':<6}"
)
print('-' * 65)

for idx, (g1, g2) in enumerate(pairs):
  adj_p_val = adj_p[idx]
  sig_tag = (
      '***'
      if adj_p_val < 0.001
      else (
          '**'
          if adj_p_val < 0.01
          else ('*' if adj_p_val < 0.05 else 'ns')
      )
  )
  p_str = '< 0.001' if adj_p_val < 0.001 else f'{adj_p_val:.3f}'
  print(
      f'{g1:<10} {g2:<10} {rrs[idx]:<12.3f} {raw_p[idx]:<10.4f} {p_str:<10}'
      f' {sig_tag:<6}'
  )