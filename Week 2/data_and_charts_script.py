import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(42)

# -----------------------------------------------------------------
# 1. BUILD A REALISTIC SYNTHETIC "GLOBAL RETAIL SALES" DATASET
#    (structured like the well-known Superstore-style retail dataset:
#     Order Date, Region, Category, Sub-Category, Segment, Sales,
#     Profit, Discount, Quantity)
# -----------------------------------------------------------------
regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"]
categories = {
    "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
    "Office Supplies": ["Binders", "Paper", "Storage", "Art"],
    "Technology": ["Phones", "Machines", "Accessories", "Copiers"],
}
segments = ["Consumer", "Corporate", "Home Office"]

n = 9000
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

rows = []
for i in range(n):
    d = rng.choice(dates)
    region = rng.choice(regions, p=[0.32, 0.24, 0.22, 0.13, 0.09])
    cat = rng.choice(list(categories.keys()), p=[0.24, 0.42, 0.34])
    subcat = rng.choice(categories[cat])
    segment = rng.choice(segments, p=[0.51, 0.31, 0.18])

    # seasonality: Nov/Dec boost (holiday shopping), mild dip in Feb
    month = pd.Timestamp(d).month
    season = 1.0
    if month in (11, 12):
        season = 1.45
    elif month == 2:
        season = 0.82
    elif month in (6, 7):
        season = 1.1

    base_price = {"Furniture": 250, "Office Supplies": 40, "Technology": 380}[cat]
    quantity = max(1, int(rng.normal(3, 1.5)))
    unit_price = max(5, rng.normal(base_price, base_price * 0.35))
    discount = float(np.clip(rng.choice([0, 0, 0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                                         p=[0.32, 0.1, 0.08, 0.15, 0.13, 0.1, 0.06, 0.04, 0.02]), 0, 0.5))
    sales = quantity * unit_price * season * (1 - discount * 0.15)  # discount slightly cuts realized sales too
    cost_ratio = {"Furniture": 0.78, "Office Supplies": 0.62, "Technology": 0.74}[cat]
    profit = sales * (1 - cost_ratio) - sales * discount * 1.6  # heavy discounts erode profit fast
    # occasional loss-making orders at high discount
    profit += rng.normal(0, sales * 0.05)

    rows.append([d, region, cat, subcat, segment, round(quantity), round(sales, 2),
                 round(profit, 2), discount])

df = pd.DataFrame(rows, columns=["OrderDate", "Region", "Category", "SubCategory",
                                  "Segment", "Quantity", "Sales", "Profit", "Discount"])
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["Month"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
df["ProfitMargin"] = df["Profit"] / df["Sales"]

df.to_csv("/home/claude/work/retail_sales_synthetic.csv", index=False)
print(df.shape)
print(df.head())
print("Total Sales: {:,.0f}".format(df.Sales.sum()))
print("Total Profit: {:,.0f}".format(df.Profit.sum()))

PALETTE = ["#2E5EAA", "#E8703A", "#3FA796", "#C74B50", "#8E6BAF"]

# -----------------------------------------------------------------
# CHART 1: Monthly Sales & Profit trend (line chart, dual narrative)
# -----------------------------------------------------------------
monthly = df.groupby("Month").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()

fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(monthly["Month"], monthly["Sales"], color=PALETTE[0], linewidth=2.8, marker="o", label="Sales")
ax1.set_ylabel("Total Sales (USD)", color=PALETTE[0], fontsize=13)
ax1.tick_params(axis="y", labelcolor=PALETTE[0])
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

ax2 = ax1.twinx()
ax2.plot(monthly["Month"], monthly["Profit"], color=PALETTE[1], linewidth=2.8, marker="s", label="Profit")
ax2.set_ylabel("Total Profit (USD)", color=PALETTE[1], fontsize=13)
ax2.tick_params(axis="y", labelcolor=PALETTE[1])
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax2.grid(False)

# annotate holiday peak
peak = monthly.loc[monthly["Sales"].idxmax()]
ax1.annotate("Holiday season peak\n(Nov\u2013Dec surge)", xy=(peak["Month"], peak["Sales"]),
             xytext=(15, 25), textcoords="offset points", fontsize=11,
             arrowprops=dict(arrowstyle="->", color="gray"))

plt.title("Monthly Sales & Profit Trend (2023\u20132024)", fontsize=16, fontweight="bold", pad=15)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("/home/claude/work/charts/01_monthly_trend.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# CHART 2: Sales by Category (bar chart)
# -----------------------------------------------------------------
cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10, 6))
bars = plt.bar(cat_sales["Category"], cat_sales["Sales"], color=PALETTE[:3], width=0.55)
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h + h*0.015, f"${h/1e6:.2f}M",
              ha="center", fontsize=12, fontweight="bold")
plt.ylabel("Total Sales (USD)")
plt.title("Total Sales by Product Category", fontsize=16, fontweight="bold", pad=15)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig("/home/claude/work/charts/02_sales_by_category.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# CHART 3: Heatmap of Sales by Region x Category
# -----------------------------------------------------------------
pivot = df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum") / 1000
plt.figure(figsize=(10, 6.5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Sales (USD, thousands)"},
            linewidths=0.5, linecolor="white")
plt.title("Sales Intensity: Region vs. Category ($K)", fontsize=16, fontweight="bold", pad=15)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
plt.savefig("/home/claude/work/charts/03_region_category_heatmap.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# CHART 4: Scatter - Discount vs Profit Margin
# -----------------------------------------------------------------
sample = df.sample(1500, random_state=1)
plt.figure(figsize=(10, 6.5))
sns.scatterplot(data=sample, x="Discount", y="ProfitMargin", hue="Category",
                 palette=PALETTE[:3], alpha=0.55, s=45)
# trend line
z = np.polyfit(df["Discount"], df["ProfitMargin"], 1)
xs = np.linspace(0, 0.5, 50)
plt.plot(xs, np.polyval(z, xs), color="black", linestyle="--", linewidth=2, label="Overall trend")
plt.axhline(0, color="gray", linewidth=1)
plt.xlabel("Discount Rate")
plt.ylabel("Profit Margin")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.title("Discount Rate vs. Profit Margin", fontsize=16, fontweight="bold", pad=15)
plt.legend(title="Category", loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("/home/claude/work/charts/04_discount_vs_margin.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# CHART 5: Box plot - Sales distribution by Segment
# -----------------------------------------------------------------
plt.figure(figsize=(10, 6.5))
sns.boxplot(data=df, x="Segment", y="Sales", palette=PALETTE[:3], showfliers=False)
plt.ylabel("Order Sales Value (USD)")
plt.xlabel("")
plt.title("Order Value Distribution by Customer Segment", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("/home/claude/work/charts/05_segment_boxplot.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# CHART 6: Stacked bar - Category share within each Region (composition)
# -----------------------------------------------------------------
comp = df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum")
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
comp_pct = comp_pct.loc[cat_sales.set_index("Category").index.name and comp_pct.sum(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(10.5, 6.5))
bottom = np.zeros(len(comp_pct))
for i, col in enumerate(comp_pct.columns):
    plt.bar(comp_pct.index, comp_pct[col], bottom=bottom, label=col, color=PALETTE[i])
    bottom += comp_pct[col].values
plt.ylabel("Share of Regional Sales (%)")
plt.title("Product Category Mix Within Each Region", fontsize=16, fontweight="bold", pad=15)
plt.xticks(rotation=20, ha="right")
plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("/home/claude/work/charts/06_regional_category_mix.png", dpi=150)
plt.close()

print("All charts saved.")

# print summary stats used for narrative
print("\n--- Narrative stats ---")
print("Category sales:\n", cat_sales)
print("\nRegion totals:\n", df.groupby('Region')['Sales'].sum().sort_values(ascending=False))
print("\nSegment median sales:\n", df.groupby('Segment')['Sales'].median())
print("\nCorrelation discount vs margin:", df['Discount'].corr(df['ProfitMargin']))
print("\nProfit margin at 0 discount vs >=0.3 discount:")
print(df[df.Discount==0]['ProfitMargin'].mean(), df[df.Discount>=0.3]['ProfitMargin'].mean())
print("\nHoliday months share of total sales:")
holiday = df[df['Month'].dt.month.isin([11,12])]['Sales'].sum()
print(holiday/df['Sales'].sum())
