import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


df = pd.read_csv("/Users/clair/Documents/BME 2315/Module 2/BME2315_Module2/Module2_FSHD_Dataset.csv")
print(df.shape)
df.head()

# Check for missing visits -- not every subject has a Year 1 and/or Year 2 visit
print("Subjects:", len(df))
print("Have a Year 1 visit:", df['lag_months_y1'].notna().sum())
print("Have a Year 2 visit:", df['lag_months_y2'].notna().sum())

T1 = df['lag_months_y1'].min()   # earliest Year-1 visit across all subjects
T2 = df['lag_months_y2'].min()   # earliest Year-2 visit across all subjects
print(f"T1 = {T1:.2f} months")
print(f"T2 = {T2:.2f} months")

row = df.loc[df['subject_id'] == '13-0004'].iloc[0]

lag_months = np.array([row['lag_months_baseline'], row['lag_months_y1'], row['lag_months_y2']])
ff_values = np.array([row['ff_baseline_R'], row['ff_y1_R'], row['ff_y2_R']])

print("lag_months:", lag_months)
print("ff_values: ", ff_values)

x1, x2 = lag_months[0], lag_months[1]
y1, y2 = ff_values[0], ff_values[1]

Z = np.array([[1, x1], [1, x2]])
Y = np.array([y1, y2])
A = np.linalg.solve(Z, Y)
a1, a2 = A

estimate_T1 = a1 + a2 * T1
print(f"Linear estimate at T1 = {T1:.2f} months: {estimate_T1:.2f}%")

Z = np.array([[1, x, x**2] for x in lag_months])
Y = ff_values
A = np.linalg.solve(Z, Y)
a1, a2, a3 = A

estimate_T2 = a1 + a2 * T2 + a3 * T2**2
print(f"Quadratic estimate at T2 = {T2:.2f} months: {estimate_T2:.2f}%")

f_linear = interp1d(lag_months, ff_values, kind='linear')
f_quadratic = interp1d(lag_months, ff_values, kind='quadratic')

print("f_linear(T1)   =", f_linear(T1))
print("f_quadratic(T2) =", f_quadratic(T2))

TEAL = "#1f7a8c"
PINK = "#e0569b"
ORANGE = "#e8821e"

x_curve = np.linspace(lag_months[0], lag_months[-1], 200)
quad_curve = f_quadratic(x_curve)

plt.figure(figsize=(7.5, 5))
plt.plot(x_curve, quad_curve, color=ORANGE, lw=2.5, linestyle="--", label="Quadratic (all 3 points)")
plt.plot(lag_months[:2], ff_values[:2], color=PINK, lw=2.5, linestyle="--", label="Linear (baseline -> Year 1)")
plt.scatter(lag_months, ff_values, color=TEAL, s=90, zorder=4, label="Measured visits (known)")
plt.scatter([T1], [estimate_T1], color=PINK, marker="D", s=120, zorder=5, label=f"Linear estimate @ T1={T1:.1f}")
plt.scatter([T2], [estimate_T2], color=ORANGE, marker="D", s=120, zorder=5, label=f"Quadratic estimate @ T2={T2:.1f}")
plt.title("Subject 13-0004 — right tibialis anterior fat fraction")
plt.xlabel("Months since baseline (lag_months)")
plt.ylabel("Fat fraction (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Get an estimate only if we have the visits we need
def get_estimate(times, values, target, kind):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(target):
        return np.nan
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        return np.nan
    if not np.all(np.diff(times) > 0):
        return np.nan
    if not times[0] <= target <= times[-1]:
        return np.nan
    return float(interp1d(times, values, kind=kind)(target))

results = []

for idx, row in df.iterrows():
    subject_id = row['subject_id']

    # TODO: pull this subject's lag_months and ff_R values into arrays,
    #       the same way we did for Subject 13-0004 above.
    lag_months = np.array([
        row['lag_months_baseline'], row['lag_months_y1'], row['lag_months_y2']
    ])
    ff_values = np.array([
        row['ff_baseline_R'], row['ff_y1_R'], row['ff_y2_R']
    ])

    # TODO: if this subject has at least baseline + Year 1 (no NaNs), compute
    #       their linear estimate at T1 (hint: interp1d with kind='linear').
    #       If they don't have enough data, use NaN instead.
    estimate_T1_linear = np.nan

    if pd.notna(lag_months[:2]).all() and pd.notna(ff_values[:2]).all():
        if lag_months[0] < lag_months[1] and lag_months[0] <= T1 <= lag_months[1]:
            f_linear = interp1d(lag_months[:2], ff_values[:2], kind='linear')
            estimate_T1_linear = float(f_linear(T1))

    # TODO: if this subject has all three visits (no NaNs), compute their
    #       quadratic estimate at T2 (hint: interp1d with kind='quadratic').
    #       If they don't, use NaN instead.
    estimate_T2_quadratic = np.nan

    if pd.notna(lag_months).all() and pd.notna(ff_values).all():
        if lag_months[0] < lag_months[1] < lag_months[2]:
            if lag_months[0] <= T2 <= lag_months[2]:
                f_quadratic = interp1d(lag_months, ff_values, kind='quadratic')
                estimate_T2_quadratic = float(f_quadratic(T2))

    # Get the clinical severity scores for this person too
    severity_values = np.array([
        row['clinical_severity_score_baseline'],
        row['clinical_severity_score_y1'],
        row['clinical_severity_score_y2']
    ])

    # Leave it as NaN if we don't have enough data
    severity_T1_linear = np.nan
    severity_T2_quadratic = np.nan

    if pd.notna(lag_months[:2]).all() and pd.notna(severity_values[:2]).all():
        if lag_months[0] < lag_months[1] and lag_months[0] <= T1 <= lag_months[1]:
            severity_linear = interp1d(
                lag_months[:2], severity_values[:2], kind='linear'
            )
            severity_T1_linear = float(severity_linear(T1))

    if pd.notna(lag_months).all() and pd.notna(severity_values).all():
        if lag_months[0] < lag_months[1] < lag_months[2]:
            if lag_months[0] <= T2 <= lag_months[2]:
                severity_quadratic = interp1d(
                    lag_months, severity_values, kind='quadratic'
                )
                severity_T2_quadratic = float(severity_quadratic(T2))

    # TODO: append a dict like
    #       {'subject_id': subject_id, 'estimate_T1_linear': ..., 'estimate_T2_quadratic': ...}
    #       to `results`
    results.append({
        'subject_id': subject_id,
        'estimate_T1_linear': estimate_T1_linear,
        'estimate_T2_quadratic': estimate_T2_quadratic,
        'severity_T1_linear': severity_T1_linear,
        'severity_T2_quadratic': severity_T2_quadratic
    })

# TODO: turn `results` into a DataFrame and inspect it
# results_df = pd.DataFrame(results)
# results_df.head()
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Check how many estimates we got for each one
print('\nNumber of estimates:')
print(results_df.count())

# Save the results like the class example
results_df.to_csv('interpolated_estimates.csv', index=False)
print('Saved interpolated_estimates.csv')

# Plot clinical severity for the same person as the fat fraction example
row = df.loc[df['subject_id'] == '13-0004'].iloc[0]
lag_months = np.array([
    row['lag_months_baseline'], row['lag_months_y1'], row['lag_months_y2']
])
severity_values = np.array([
    row['clinical_severity_score_baseline'],
    row['clinical_severity_score_y1'],
    row['clinical_severity_score_y2']
])

# This person has all three scores, so we can show both estimates
if pd.notna(lag_months).all() and pd.notna(severity_values).all():
    if lag_months[0] < lag_months[1] < lag_months[2]:
        if lag_months[0] <= T1 <= lag_months[1] and lag_months[0] <= T2 <= lag_months[2]:
            severity_linear = interp1d(
                lag_months[:2], severity_values[:2], kind='linear'
            )
            severity_quadratic = interp1d(
                lag_months, severity_values, kind='quadratic'
            )
            severity_estimate_T1 = float(severity_linear(T1))
            severity_estimate_T2 = float(severity_quadratic(T2))

            print(f'Clinical severity at T1: {severity_estimate_T1:.2f}')
            print(f'Clinical severity at T2: {severity_estimate_T2:.2f}')

            x_curve = np.linspace(lag_months[0], lag_months[-1], 200)
            severity_curve = severity_quadratic(x_curve)

            plt.figure(figsize=(7.5, 5))
            plt.plot(x_curve, severity_curve, color=ORANGE, linestyle='--',
                     label='Quadratic (all 3 points)')
            plt.plot(lag_months[:2], severity_values[:2], color=PINK,
                     linestyle='--', label='Linear (baseline -> Year 1)')
            plt.scatter(lag_months, severity_values, color=TEAL, s=90,
                        zorder=4, label='Measured visits (known)')
            plt.scatter([T1], [severity_estimate_T1], color=PINK, marker='D',
                        s=120, zorder=5, label=f'Linear estimate @ T1={T1:.1f}')
            plt.scatter([T2], [severity_estimate_T2], color=ORANGE, marker='D',
                        s=120, zorder=5, label=f'Quadratic estimate @ T2={T2:.1f}')
            plt.title('Subject 13-0004 — clinical severity')
            plt.xlabel('Months since baseline (lag_months)')
            plt.ylabel('Clinical severity score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
else:
    print('Not enough data for the clinical severity example plot.')

    # Compare changes in fat fraction and clinical severity for the same people
from scipy.stats import spearmanr

correlation_results = []
# Baseline to Year 1
paired = df[['subject_id', 'ff_baseline_R', 'ff_y1_R',
             'clinical_severity_score_baseline', 'clinical_severity_score_y1',
             'lag_months_baseline', 'lag_months_y1']].dropna().copy()

paired['ff_change'] = paired['ff_y1_R'] - paired['ff_baseline_R']
paired['severity_change'] = (
    paired['clinical_severity_score_y1'] - paired['clinical_severity_score_baseline']
)
n = len(paired)
rho = np.nan
p_approx = np.nan
p_shuffle = np.nan

# We need some variation in both changes to calculate a correlation
if n >= 3 and paired['ff_change'].nunique() > 1 and paired['severity_change'].nunique() > 1:
    rho, p_approx = spearmanr(paired['ff_change'], paired['severity_change'])

    # Shuffle the pairings to check the p-value with this small group
    # The seed makes the answer repeatable when someone else runs it
    rng = np.random.default_rng(42)
    extreme = 0
    for repeat in range(9999):
        shuffled = rng.permutation(paired['severity_change'].to_numpy())
        shuffled_rho, unused_p = spearmanr(paired['ff_change'], shuffled)
        if abs(shuffled_rho) >= abs(rho) - 1e-12:
            extreme = extreme + 1
    p_shuffle = (extreme + 1) / 10000
else:
    print('Not enough data or variation for the Year 1 correlation.')

print('\nBaseline to Year 1:')
print('Participants with both changes:', n)
print('Spearman rho:', rho)
print('Approximate p-value:', p_approx)
print('Two-sided shuffled p-value:', p_shuffle)
print('Follow-up months (minimum, median, maximum):',
      paired['lag_months_y1'].min(), paired['lag_months_y1'].median(), paired['lag_months_y1'].max())

correlation_results.append({
    'comparison': 'Baseline to Year 1',
    'n_pairs': n,
    'spearman_rho': rho,
    'p_approx': p_approx,
    'p_permutation': p_shuffle,
    'followup_months_min': paired['lag_months_y1'].min(),
    'followup_months_median': paired['lag_months_y1'].median(),
    'followup_months_max': paired['lag_months_y1'].max()
})
paired.to_csv('changes_year1.csv', index=False)

# Each dot is one person with both changes available
plt.figure(figsize=(8, 5.5))
plt.scatter(paired['ff_change'], paired['severity_change'],
            color=TEAL, s=70, alpha=0.75)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title(f'Do fat fraction and clinical severity change together?\n'
          f'Baseline to Year 1 | {n} participants')
plt.xlabel('Change in right TA fat fraction (percentage points)')
plt.ylabel('Change in clinical severity score (points)')
plt.text(0.03, 0.97, f'Spearman rho = {rho:.2f}\nPermutation p = {p_shuffle:.3f}',
         transform=plt.gca().transAxes, va='top',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgray'))
plt.grid(alpha=0.2)
plt.figtext(0.5, 0.01, 'Change = follow-up minus baseline. Each dot represents one participant; dots may overlap.',
            ha='center', fontsize=8)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
# Baseline to Year 2
paired = df[['subject_id', 'ff_baseline_R', 'ff_y2_R',
             'clinical_severity_score_baseline', 'clinical_severity_score_y2',
             'lag_months_baseline', 'lag_months_y2']].dropna().copy()

paired['ff_change'] = paired['ff_y2_R'] - paired['ff_baseline_R']
paired['severity_change'] = (
    paired['clinical_severity_score_y2'] - paired['clinical_severity_score_baseline']
)
n = len(paired)
rho = np.nan
p_approx = np.nan
p_shuffle = np.nan

# We need some variation in both changes to calculate a correlation
if n >= 3 and paired['ff_change'].nunique() > 1 and paired['severity_change'].nunique() > 1:
    rho, p_approx = spearmanr(paired['ff_change'], paired['severity_change'])

    # Shuffle the pairings to check the p-value with this small group
    # The seed makes the answer repeatable when someone else runs it
    rng = np.random.default_rng(42)
    extreme = 0
    for repeat in range(9999):
        shuffled = rng.permutation(paired['severity_change'].to_numpy())
        shuffled_rho, unused_p = spearmanr(paired['ff_change'], shuffled)
        if abs(shuffled_rho) >= abs(rho) - 1e-12:
            extreme = extreme + 1
    p_shuffle = (extreme + 1) / 10000
else:
    print('Not enough data or variation for the Year 2 correlation.')

print('\nBaseline to Year 2:')
print('Participants with both changes:', n)
print('Spearman rho:', rho)
print('Approximate p-value:', p_approx)
print('Two-sided shuffled p-value:', p_shuffle)
print('Follow-up months (minimum, median, maximum):',
      paired['lag_months_y2'].min(), paired['lag_months_y2'].median(), paired['lag_months_y2'].max())

correlation_results.append({
    'comparison': 'Baseline to Year 2',
    'n_pairs': n,
    'spearman_rho': rho,
    'p_approx': p_approx,
    'p_permutation': p_shuffle,
    'followup_months_min': paired['lag_months_y2'].min(),
    'followup_months_median': paired['lag_months_y2'].median(),
    'followup_months_max': paired['lag_months_y2'].max()
})
paired.to_csv('changes_year2.csv', index=False)

# Each dot is one person with both changes available
plt.figure(figsize=(8, 5.5))
plt.scatter(paired['ff_change'], paired['severity_change'],
            color=TEAL, s=70, alpha=0.75)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title(f'Do fat fraction and clinical severity change together?\n'
          f'Baseline to Year 2 | {n} participants')
plt.xlabel('Change in right TA fat fraction (percentage points)')
plt.ylabel('Change in clinical severity score (points)')
plt.text(0.03, 0.97, f'Spearman rho = {rho:.2f}\nPermutation p = {p_shuffle:.3f}',
         transform=plt.gca().transAxes, va='top',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgray'))
plt.grid(alpha=0.2)
plt.figtext(0.5, 0.01, 'Change = follow-up minus baseline. Each dot represents one participant; dots may overlap.',
            ha='center', fontsize=8)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
# We did two comparisons, so show corrected p-values too
correlation_df = pd.DataFrame(correlation_results)
correlation_df['p_bonferroni'] = correlation_df['p_permutation'] * 2
correlation_df.loc[correlation_df['p_bonferroni'] > 1, 'p_bonferroni'] = 1
print('\nCorrelation results:')
print(correlation_df.to_string(index=False))
correlation_df.to_csv('spearman_results.csv', index=False)
print('Saved spearman_results.csv')
