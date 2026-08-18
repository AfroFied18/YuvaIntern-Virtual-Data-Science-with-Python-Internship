import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
PALETTE = ["#2E5EAA", "#E8703A", "#3FA796", "#C74B50", "#8E6BAF"]

df = pd.read_csv("retail_sales_synthetic.csv")
df["Discounted"] = np.where(df["Discount"] > 0, "Discounted", "No Discount")

print("Rows:", len(df))
print(df[["Sales","Profit","Discount","ProfitMargin"]].describe())

# ============================================================
# TEST 1: Independent samples t-test
# H0: mean profit margin (discounted) = mean profit margin (no discount)
# H1: mean profit margin (discounted) < mean profit margin (no discount)
# ============================================================
disc = df.loc[df.Discounted == "Discounted", "ProfitMargin"]
nodisc = df.loc[df.Discounted == "No Discount", "ProfitMargin"]

t_stat, p_two = stats.ttest_ind(disc, nodisc, equal_var=False)
p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2  # one-tailed, testing "less than"

mean_diff = disc.mean() - nodisc.mean()
se_diff = np.sqrt(disc.var(ddof=1)/len(disc) + nodisc.var(ddof=1)/len(nodisc))
ci_low, ci_high = mean_diff - 1.96*se_diff, mean_diff + 1.96*se_diff

print("\n--- T-TEST: Discounted vs Non-Discounted Profit Margin ---")
print(f"n(discounted)={len(disc)}, mean={disc.mean():.4f}, sd={disc.std():.4f}")
print(f"n(no discount)={len(nodisc)}, mean={nodisc.mean():.4f}, sd={nodisc.std():.4f}")
print(f"Mean difference = {mean_diff:.4f}")
print(f"t = {t_stat:.3f}, two-tailed p = {p_two:.6g}, one-tailed p = {p_one:.6g}")
print(f"95% CI of difference: [{ci_low:.4f}, {ci_high:.4f}]")

# ============================================================
# TEST 2: Chi-square test of independence
# H0: Product Category and Customer Segment are independent
# H1: they are NOT independent (segment influences category preference)
# ============================================================
contingency = pd.crosstab(df["Segment"], df["Category"])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

n_total = contingency.values.sum()
phi2 = chi2 / n_total
r, k = contingency.shape
cramers_v = np.sqrt(phi2 / (min(r-1, k-1)))

print("\n--- CHI-SQUARE TEST: Segment vs Category ---")
print(contingency)
print(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p_chi:.6g}")
print(f"Cramer's V = {cramers_v:.4f}")

# ============================================================
# TEST 3: One-way ANOVA
# H0: mean Sales is equal across all 5 regions
# H1: at least one region's mean Sales differs
# ============================================================
groups = [g["Sales"].values for _, g in df.groupby("Region")]
region_labels = df["Region"].unique().tolist()
f_stat, p_anova = stats.f_oneway(*groups)

# manual ANOVA table (sum of squares) for transparency
grand_mean = df["Sales"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
ss_within = sum(((g - g.mean())**2).sum() for g in groups)
df_between = len(groups) - 1
df_within = len(df) - len(groups)
ms_between = ss_between / df_between
ms_within = ss_within / df_within
f_manual = ms_between / ms_within

print("\n--- ONE-WAY ANOVA: Sales across Regions ---")
region_means = df.groupby("Region")["Sales"].agg(["count","mean","std"])
print(region_means)
print(f"SS_between={ss_between:.1f}, SS_within={ss_within:.1f}")
print(f"df_between={df_between}, df_within={df_within}")
print(f"F = {f_manual:.3f} (scipy F = {f_stat:.3f}), p = {p_anova:.6g}")

# post-hoc: pairwise t-tests with Bonferroni-style flag (just for narrative, not full Tukey)
na = df.loc[df.Region=="North America","Sales"]
mea = df.loc[df.Region=="Middle East & Africa","Sales"]
t2, p2 = stats.ttest_ind(na, mea, equal_var=False)
print(f"\nPost-hoc check, North America vs Middle East & Africa: t={t2:.3f}, p={p2:.6g}")

# ============================================================
# VISUALIZATIONS
# ============================================================

# Chart 1: Boxplot of profit margin by discount group
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Discounted", y="ProfitMargin", palette=PALETTE[:2], showfliers=False)
plt.title("Profit Margin: Discounted vs Non-Discounted Orders")
plt.ylabel("Profit Margin")
plt.xlabel("")
plt.tight_layout()
plt.savefig("charts/t1_boxplot_discount.png", dpi=150)
plt.close()

# Chart 2: Overlaid histograms
plt.figure(figsize=(9,6))
plt.hist(nodisc, bins=40, alpha=0.6, label="No Discount", color=PALETTE[0], density=True)
plt.hist(disc, bins=40, alpha=0.6, label="Discounted", color=PALETTE[1], density=True)
plt.axvline(nodisc.mean(), color=PALETTE[0], linestyle="--", linewidth=2)
plt.axvline(disc.mean(), color=PALETTE[1], linestyle="--", linewidth=2)
plt.xlabel("Profit Margin")
plt.ylabel("Density")
plt.title("Distribution of Profit Margin by Discount Status")
plt.legend()
plt.tight_layout()
plt.savefig("charts/t2_histogram_margin.png", dpi=150)
plt.close()

# Chart 3: Contingency heatmap for chi-square
plt.figure(figsize=(8,6))
sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", cbar_kws={"label":"Order count"})
plt.title("Order Counts: Customer Segment vs Product Category")
plt.ylabel("Segment")
plt.xlabel("Category")
plt.tight_layout()
plt.savefig("charts/t3_chisquare_heatmap.png", dpi=150)
plt.close()

# Chart 4: Boxplot Sales by region for ANOVA
plt.figure(figsize=(10,6.5))
order = df.groupby("Region")["Sales"].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x="Region", y="Sales", order=order, palette=PALETTE, showfliers=False)
plt.xticks(rotation=20, ha="right")
plt.title("Sales Distribution by Region (ANOVA)")
plt.ylabel("Order Sales (USD)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("charts/t4_anova_boxplot.png", dpi=150)
plt.close()

# Chart 5: Group means with 95% CI error bars
plt.figure(figsize=(10,6.5))
means = df.groupby("Region")["Sales"].mean().reindex(order)
sems = df.groupby("Region")["Sales"].sem().reindex(order)
ci95 = sems * 1.96
plt.bar(means.index, means.values, yerr=ci95.values, capsize=6, color=PALETTE)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Mean Order Sales (USD)")
plt.title("Regional Mean Sales with 95% Confidence Intervals")
plt.tight_layout()
plt.savefig("charts/t5_region_means_ci.png", dpi=150)
plt.close()

print("\nAll charts saved to charts/")
