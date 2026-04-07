"""
DAT1 - Boston Housing Dataset
Ethan Zhuo

Column info:
1)  age_years         - Age of house (years)
2)  sqft              - Square footage
3)  rooms             - Total rooms
4)  bedrooms          - Number of bedrooms
5)  bathrooms         - Number of bathrooms
6)  pool              - Has pool (0/1)
7)  garage            - Has garage (0/1)
8)  zip_median_income - Median income in zipcode ($)
9)  dist_transit_ft   - Distance to nearest transit (ft)
10) trust_score       - Community trust score (1-10)
11) house_value       - Assessed house value ($) -- OUTCOME
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv('bostonHousingDataset.csv')

X_cols = ['age_years', 'sqft', 'rooms', 'bedrooms', 'bathrooms',
          'pool', 'garage', 'zip_median_income', 'dist_transit_ft', 'trust_score']
y      = df['house_value'].values
y_mean = y.mean()
ss_tot = np.sum((y - y_mean) ** 2)


def ols(Xmat, yarr):
    """OLS via the Normal Equation. Returns betas, y_pred, R2, RMSE."""
    Xb            = np.column_stack([np.ones(len(Xmat)), Xmat])
    betas, _, _, _ = lstsq(Xb, yarr, rcond=None)
    yp    = Xb @ betas
    r2    = 1 - np.sum((yarr - yp) ** 2) / ss_tot
    rmse  = np.sqrt(np.mean((yarr - yp) ** 2))
    return betas, yp, r2, rmse



# QUESTION 1 – Best single predictor of house value

print("Q1: Simple Linear Regression – Each Predictor vs. House Value")

q1 = {}
for col in X_cols:
    x = df[col].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2   = r ** 2
    yp   = slope * x + intercept
    rmse = np.sqrt(np.mean((y - yp) ** 2))
    q1[col] = dict(r=r, r2=r2, slope=slope, intercept=intercept, rmse=rmse, p=p)
    print(f"  {col:<22}  r = {r:+.3f}  R² = {r2:.3f}  RMSE = ${rmse:,.0f}  p = {p:.2e}")

ranked     = sorted(q1.items(), key=lambda x: x[1]['r2'], reverse=True)
best_var   = ranked[0][0]
worst_var  = ranked[-1][0]
print(f"\nBest predictor:  {best_var}  (R² = {q1[best_var]['r2']:.3f})")
print(f"Worst predictor: {worst_var}  (R² = {q1[worst_var]['r2']:.3f})")

#Figure Q1: scatter plots for all 10 predictors
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()
for i, col in enumerate(X_cols):
    ax  = axes[i]
    x   = df[col].values
    sl  = q1[col]['slope']
    ic  = q1[col]['intercept']
    r2  = q1[col]['r2']
    r   = q1[col]['r']
    ax.scatter(x, y, alpha=0.2, s=4, color='steelblue')
    xln = np.linspace(x.min(), x.max(), 300)
    ax.plot(xln, sl * xln + ic, 'r-', lw=1.5, label=f'R² = {r2:.3f}, r = {r:+.3f}')
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel('House Value ($)', fontsize=7)
    ax.set_title(col, fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}k'))
plt.suptitle('Q1 – Simple Linear Regression: Each Predictor vs. House Value',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: q1_scatter.png")


# QUESTION 2 – Multiple regression with all 10 predictors
print("\n")
print("\n")
print("\n")
print("Q2: Multiple Regression – All 10 Predictors")

b_full, yp_full, r2_full, rmse_full = ols(df[X_cols].values, y)

print(f"R²   = {r2_full:.3f}")
print(f"RMSE = ${rmse_full:,.0f}")
print(f"Intercept (β0) = {b_full[0]:,.0f}")
for i, col in enumerate(X_cols):
    print(f"  β_{col:<22} = {b_full[i+1]:.3f}")

r2_best = q1[best_var]['r2']
gain    = r2_full - r2_best
print(f"\nBest single predictor ({best_var}): R² = {r2_best:.3f}")
print(f"Full model:                        R² = {r2_full:.3f}  (+{gain:.3f}, {gain*100:.1f} pp)")

#Figure Q2: predicted vs. actual
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(yp_full, y, alpha=0.2, s=5, color='steelblue', label='Observations')
lo = min(yp_full.min(), y.min()); hi = max(yp_full.max(), y.max())
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = ŷ (perfect)')
ax.set_xlabel('Predicted House Value (ŷ)', fontsize=10)
ax.set_ylabel('Actual House Value (y)', fontsize=10)
ax.set_title(f'Q2 – Predicted vs. Actual (All 10 Predictors)\nR² = {r2_full:.3f}', fontsize=10)
ax.legend(fontsize=9)
fmt = plt.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M' if abs(v) >= 1e6 else f'${v/1e3:.0f}k')
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
plt.tight_layout()
plt.savefig('q2_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: q2_pred_vs_actual.png")


# QUESTION 3 – Collinearity check; build best interpretable model
print("\n")
print("\n")
print("\n")
print("Q3: Collinearity Check and Reduced Model")

corr_full = df[X_cols].corr()
print("\nPairs with |r| > 0.70 (serious collinearity):")
high_pairs = []
for i in range(len(X_cols)):
    for j in range(i + 1, len(X_cols)):
        rv = corr_full.iloc[i, j]
        if abs(rv) > 0.70:
            high_pairs.append((X_cols[i], X_cols[j], rv))
            print(f"  {X_cols[i]} & {X_cols[j]:<22}  r = {rv:.3f}")

# Drop the lower-R² member of each collinear pair
to_drop = set()
for c1, c2, _ in high_pairs:
    drop = c2 if q1[c1]['r2'] >= q1[c2]['r2'] else c1
    keep = c1 if drop == c2 else c2
    if drop not in to_drop:
        to_drop.add(drop)
        print(f"  --> drop '{drop}' (R²={q1[drop]['r2']:.3f}), keep '{keep}' (R²={q1[keep]['r2']:.3f})")

red_cols = [c for c in X_cols if c not in to_drop]

# For any remaining pair above 0.70
corr_red  = df[red_cols].corr()
still_bad = [(red_cols[i], red_cols[j], corr_red.iloc[i, j])
             for i in range(len(red_cols)) for j in range(i+1, len(red_cols))
             if abs(corr_red.iloc[i, j]) > 0.70]
while still_bad:
    c1, c2, _ = still_bad[0]
    drop = c2 if q1[c1]['r2'] >= q1[c2]['r2'] else c1
    red_cols.remove(drop)
    corr_red  = df[red_cols].corr()
    still_bad = [(red_cols[i], red_cols[j], corr_red.iloc[i, j])
                 for i in range(len(red_cols)) for j in range(i+1, len(red_cols))
                 if abs(corr_red.iloc[i, j]) > 0.70]

print(f"\nReduced set ({len(red_cols)} vars): {red_cols}")

b_red, yp_red, r2_red, rmse_red = ols(df[red_cols].values, y)
print(f"Reduced model: R² = {r2_red:.3f}  RMSE = ${rmse_red:,.0f}")
print(f"Intercept (β0) = {b_red[0]:,.0f}")
for i, col in enumerate(red_cols):
    print(f"  β_{col:<22} = {b_red[i+1]:.3f}")

print("\nAll pairwise |r| in reduced set (must be < 0.70):")
for i in range(len(red_cols)):
    for j in range(i + 1, len(red_cols)):
        rv = corr_red.iloc[i, j]
        print(f"  {red_cols[i]} & {red_cols[j]:<22}  r = {rv:.3f}")

# Figure Q3a: correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_full.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(X_cols))); ax.set_yticks(range(len(X_cols)))
ax.set_xticklabels(X_cols, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(X_cols, fontsize=8)
for i in range(len(X_cols)):
    for j in range(len(X_cols)):
        val   = corr_full.iloc[i, j]
        ctxt  = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=ctxt)
for c1, c2, _ in high_pairs:
    ri = X_cols.index(c1); ci = X_cols.index(c2)
    for (r_, c_) in [(ri, ci), (ci, ri)]:
        rect = plt.Rectangle((c_ - 0.5, r_ - 0.5), 1, 1,
                              lw=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
ax.set_title('Q3 – Pairwise Correlation Matrix (red boxes: |r| > 0.70)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

#Figure Q3b: predicted vs. actual (reduced model)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(yp_red, y, alpha=0.2, s=5, color='steelblue', label='Observations')
lo = min(yp_red.min(), y.min()); hi = max(yp_red.max(), y.max())
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = ŷ (perfect)')
ax.set_xlabel('Predicted House Value (ŷ)', fontsize=10)
ax.set_ylabel('Actual House Value (y)', fontsize=10)
ax.set_title(f'Q3 – Predicted vs. Actual (Reduced Model, 7 Predictors)\nR² = {r2_red:.3f}', fontsize=10)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
plt.tight_layout()
plt.savefig('q3_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figures saved: q3_heatmap.png, q3_pred_vs_actual.png")



# QUESTION 4 – Pool vs. Garage (using Q3 reduced model coefficients)
print("\n")
print("\n")
print("\n")
print("Q4: Pool vs Garage – Impact on House Value")

b_pool   = b_red[red_cols.index('pool')   + 1]
b_garage = b_red[red_cols.index('garage') + 1]
print(f"β_pool   = ${b_pool:,.0f}  (holding all other predictors constant)")
print(f"β_garage = ${b_garage:,.0f}  (holding all other predictors constant)")
bigger   = 'pool' if abs(b_pool) > abs(b_garage) else 'garage'
print(f"Higher impact: {bigger}")

pool_yes = y[df['pool']   == 1].mean()
pool_no  = y[df['pool']   == 0].mean()
gar_yes  = y[df['garage'] == 1].mean()
gar_no   = y[df['garage'] == 0].mean()

# Figure Q4: bar chart 
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
for ax, feat, bval, yes_v, no_v in [
    (axes[0], 'Pool',   b_pool,   pool_yes, pool_no),
    (axes[1], 'Garage', b_garage, gar_yes,  gar_no),
]:
    bars = ax.bar(['Without', 'With'], [no_v / 1e3, yes_v / 1e3],
                  color=['#6baed6', '#2171b5'], edgecolor='black', width=0.5)
    ax.set_title(f'{feat}  (β = ${bval:,.0f})', fontsize=10)
    ax.set_ylabel('Mean House Value ($1,000s)', fontsize=9)
    ax.set_xlabel(f'Has {feat}?', fontsize=9)
    for bar, val in zip(bars, [no_v, yes_v]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'${val/1e3:.0f}k', ha='center', va='bottom', fontsize=9)
plt.suptitle('Q4 – Pool vs. Garage: Impact on House Value', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q4_pool_vs_garage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: q4_pool_vs_garage.png")


# QUESTION 5 – Problematic variables
print("\n")
print("\n")
print("\n")
print("Q5: Problematic Variables")

print(f"\n{'Variable':<22}  {'Skewness':>9}  {'Kurtosis':>9}")
for col in X_cols + ['house_value']:
    print(f"  {col:<22}  {df[col].skew():>9.3f}  {df[col].kurtosis():>9.3f}")

print(f"\nPool value counts: {df['pool'].value_counts().to_dict()}")
print(f"Pool = 1 in only {df['pool'].mean()*100:.1f}% of observations (severe class imbalance)")

#Figure Q5: histograms of all variables
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
axes = axes.flatten()
all_cols = X_cols + ['house_value']
for i, col in enumerate(all_cols):
    ax = axes[i]
    ax.hist(df[col], bins=40, color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_title(f'{col}\nskew = {df[col].skew():.2f}', fontsize=7, fontweight='bold')
    ax.set_xlabel(col, fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(labelsize=6)
for i in range(len(all_cols), len(axes)):
    axes[i].set_visible(False)
plt.suptitle('Q5 – Distributions of All Variables', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q5_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: q5_distributions.png")


# EXTRA CREDIT a – Normality
print("\n")
print("\n")
print("\n")
print("Extra Credit a: Normality Check")
for col in X_cols + ['house_value']:
    s = df[col].skew()
    k = df[col].kurtosis()
    print(f"  {col:<22}  skew = {s:.3f}  kurtosis = {k:.3f}")



# EXTRA CREDIT b – 2nd best (vars 1-7) vs. neighborhood factors (vars 8-10)
print("\n")
print("\n")
print("\n")
print("Extra Credit b: 2nd best Individual Predictor vs. Neighborhood Model")

vars_1to7      = ['age_years','sqft','rooms','bedrooms','bathrooms','pool','garage']
ranked_1to7    = sorted(vars_1to7, key=lambda c: q1[c]['r2'], reverse=True)
second_best    = ranked_1to7[1]
r2_second      = q1[second_best]['r2']
rmse_second    = q1[second_best]['rmse']
print(f"Variables 1-7 ranked: {[(c, round(q1[c]['r2'],3)) for c in ranked_1to7]}")
print(f"2nd best (vars 1-7): {second_best}  R² = {r2_second:.3f}  RMSE = ${rmse_second:,.0f}")

neigh_cols = ['zip_median_income', 'dist_transit_ft', 'trust_score']
_, _, r2_neigh, rmse_neigh = ols(df[neigh_cols].values, y)
print(f"Neighborhood model (vars 8-10): R² = {r2_neigh:.3f}  RMSE = ${rmse_neigh:,.0f}")
print(f"Winner: {'2nd best (' + second_best + ')' if r2_second > r2_neigh else 'neighborhood factors'}")

print("\n")
print("The End.")
