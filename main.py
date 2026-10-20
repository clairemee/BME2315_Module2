import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known points
x1, x2 = 1, 3
y1, y2 = 1, 3

# Z and Y matrices
Z = np.array([[1, x1],
              [1, x2]])
Y = np.array([y1, y2])

# Solve for A = [a1, a2]
A = np.linalg.solve(Z, Y)
print("A =", A)

# Use the coefficients to estimate y at a new x
a1, a2 = A
x_value = 2
y_value = a1 + a2 * x_value
print(f"At x = {x_value}, y = {y_value}")

# Plot the line and the interpolated point
x_line = np.linspace(0, 5, 100)
y_line = a1 + a2 * x_line

plt.figure(figsize=(6, 4))
plt.plot(x_line, y_line, label="Linear interpolating function")
plt.scatter([x1, x2], [y1, y2], color="tab:blue", s=70, zorder=4, label="Known points")
plt.scatter([x_value], [y_value], color="tab:red", s=90, zorder=5,
            label=f"Interpolated point ({x_value}, {y_value:.2f})")
plt.title("Linear interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

# Known points
x1, x2, x3 = 1, 3, 5
y1, y2, y3 = 1, 3, 2

Z = np.array([[1, x1, x1**2],
              [1, x2, x2**2],
              [1, x3, x3**2]])
Y = np.array([y1, y2, y3])

A = np.linalg.solve(Z, Y)
print("A =", A)

a1, a2, a3 = A
x_value = 4
y_value = a1 + a2 * x_value + a3 * x_value**2
print(f"At x = {x_value}, y = {y_value}")

x_curve = np.linspace(0, 6, 200)
y_curve = a1 + a2 * x_curve + a3 * x_curve**2

plt.figure(figsize=(6, 4))
plt.plot(x_curve, y_curve, color="tab:orange", label="Quadratic interpolating function")
plt.scatter([x1, x2, x3], [y1, y2, y3], color="tab:blue", s=70, zorder=4, label="Known points")
plt.scatter([x_value], [y_value], color="tab:red", s=90, zorder=5,
            label=f"Interpolated point ({x_value}, {y_value:.2f})")
plt.title("Quadratic interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

df = pd.read_csv("Module2_FSHD_Dataset.csv")
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

results = []

for idx, row in df.iterrows():
    subject_id = row['subject_id']

    # TODO: pull this subject's lag_months and ff_R values into arrays,
    #       the same way we did for Subject 13-0004 above.

    # TODO: if this subject has at least baseline + Year 1 (no NaNs), compute
    #       their linear estimate at T1 (hint: interp1d with kind='linear').
    #       If they don't have enough data, use NaN instead.

    # TODO: if this subject has all three visits (no NaNs), compute their
    #       quadratic estimate at T2 (hint: interp1d with kind='quadratic').
    #       If they don't, use NaN instead.

    # TODO: append a dict like
    #       {'subject_id': subject_id, 'estimate_T1_linear': ..., 'estimate_T2_quadratic': ...}
    #       to `results`
    pass

# TODO: turn `results` into a DataFrame and inspect it
# results_df = pd.DataFrame(results)
# results_df.head()
