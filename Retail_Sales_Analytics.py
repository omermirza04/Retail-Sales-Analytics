# -*- coding: utf-8 -*-
"""Retail Sales, Profitability & Business Performance Analysis

One-file version combining Phase 1 Assignments 1-4:
1. Load and inspect the raw retail dataset
2. Audit data quality and clean the data
3. Calculate business KPIs and investigate sales, products, customers,
   profitability, and returns
4. Create five portfolio-ready charts

HOW TO RUN
- Put this script in the same folder as:
  Phase_1_Retail_Sales_Analytics_Dataset.xlsx
- Install requirements if needed: pip install pandas numpy matplotlib openpyxl
- Run: python Retail_Sales_Analytics_Complete.py

Outputs are saved in ./retail_sales_outputs/
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. SETTINGS AND INPUT
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "retail_sales_outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
for folder in (OUTPUT_DIR, CHART_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RAW_EXCEL = BASE_DIR / "Phase_1_Retail_Sales_Analytics_Dataset.xlsx"
CLEANED_CSV = BASE_DIR / "cleaned_retail_sales.csv"

REQUIRED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name", "Region",
    "Product_Category", "Product", "Quantity", "Unit_Price", "Unit_Cost",
    "Discount", "Sales_Channel", "Payment_Method", "Return_Flag",
    "Shipping_Days"
]

if RAW_EXCEL.exists():
    print(f"Loading raw Excel dataset: {RAW_EXCEL.name}")
    df = pd.read_excel(RAW_EXCEL)
elif CLEANED_CSV.exists():
    print(f"Raw Excel not found; loading existing CSV: {CLEANED_CSV.name}")
    df = pd.read_csv(CLEANED_CSV)
    warnings.warn(
        "Loaded an existing CSV. If it is already cleaned, cleaning will run again; "
        "for a fully reproducible project, keep the original Excel file beside this script."
    )
else:
    raise FileNotFoundError(
        "Dataset not found. Put 'Phase_1_Retail_Sales_Analytics_Dataset.xlsx' "
        "in the same folder as this script."
    )

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Keep an untouched snapshot for before/after quality comparisons.
raw_df = df.copy()
print("\n" + "=" * 72)
print("SECTION 1 — INITIAL DATA EXPLORATION")
print("=" * 72)
print(f"Initial shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nFirst five rows:")
print(df.head().to_string(index=False))
print("\nColumn data types before cleaning:")
print(df.dtypes.to_string())

initial_missing = pd.DataFrame({
    "Missing_Count": raw_df.isna().sum(),
    "Missing_Percent": (raw_df.isna().mean() * 100).round(2),
})
print("\nInitial missing-value audit:")
print(initial_missing[initial_missing["Missing_Count"] > 0].to_string())
print(f"\nExact duplicate rows before cleaning: {raw_df.duplicated().sum():,}")

# -----------------------------------------------------------------------------
# 2. DATA QUALITY AUDIT AND CLEANING
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 2 — DATA QUALITY AUDIT AND CLEANING")
print("=" * 72)

# Convert dates and numeric columns safely.
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
for col in ["Quantity", "Unit_Price", "Unit_Cost", "Discount", "Shipping_Days"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Capture the initial issues before applying cleaning actions.
quality_counts = {
    "Exact duplicate rows": int(df.duplicated().sum()),
    "Invalid/unparseable dates": int(df["Order_Date"].isna().sum()),
    "Negative quantities": int((df["Quantity"] < 0).sum()),
    "Missing/zero/negative unit price": int((df["Unit_Price"].isna() | (df["Unit_Price"] <= 0)).sum()),
    "Missing/zero/negative unit cost": int((df["Unit_Cost"].isna() | (df["Unit_Cost"] <= 0)).sum()),
    "Missing discounts": int(df["Discount"].isna().sum()),
    "Missing shipping days": int(df["Shipping_Days"].isna().sum()),
    "Discount below 0 or above 1": int(((df["Discount"] < 0) | (df["Discount"] > 1)).sum()),
}
print("Issue counts before cleaning:")
for issue, count in quality_counts.items():
    print(f"- {issue}: {count:,}")

# Remove exact duplicate rows.
rows_before = len(df)
df = df.drop_duplicates().copy()
duplicates_removed = rows_before - len(df)
print(f"\nRemoved {duplicates_removed:,} exact duplicate rows.")

# Standardize text columns. This combines values differing only by extra spaces/case.
def clean_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    result = result.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return result

text_columns = [
    "Order_ID", "Customer_ID", "Customer_Name", "Region", "Product_Category",
    "Product", "Sales_Channel", "Payment_Method", "Return_Flag"
]
for col in text_columns:
    df[col] = clean_text(df[col])

# Use consistent title case for descriptive dimensions (not IDs).
for col in ["Customer_Name", "Region", "Product_Category", "Product", "Sales_Channel", "Payment_Method"]:
    df[col] = df[col].str.title()

# Missing labels are explicitly identified instead of silently dropped in groupings.
for col in ["Customer_Name", "Region", "Product_Category", "Product", "Sales_Channel", "Payment_Method"]:
    df[col] = df[col].fillna("Unknown")

# Negative quantities are retained as returns; mark them accordingly.
negative_quantity_mask = df["Quantity"] < 0
df.loc[negative_quantity_mask, "Return_Flag"] = "Yes"
df["Return_Flag"] = df["Return_Flag"].fillna("Unknown").str.strip().str.title()

# Replace missing, zero, or negative price/cost with product median, then global median.
for col in ["Unit_Price", "Unit_Cost"]:
    df.loc[df[col] <= 0, col] = np.nan
    product_median = df.groupby("Product")[col].transform("median")
    global_median = df[col].median()
    df[col] = df[col].fillna(product_median).fillna(global_median)

# Fill missing discount with 0 (no discount recorded); shipping days with median.
df["Discount"] = df["Discount"].fillna(0)
shipping_median = df["Shipping_Days"].median()
df["Shipping_Days"] = df["Shipping_Days"].fillna(shipping_median if pd.notna(shipping_median) else 0)

# Recalculate all financial metrics consistently.
df["Gross_Revenue"] = df["Quantity"] * df["Unit_Price"]
df["Discount_Amount"] = df["Gross_Revenue"] * df["Discount"]
df["Net_Revenue"] = df["Gross_Revenue"] - df["Discount_Amount"]
df["Total_Cost"] = df["Quantity"] * df["Unit_Cost"]
df["Gross_Profit"] = df["Net_Revenue"] - df["Total_Cost"]
df["Gross_Margin"] = np.where(
    df["Net_Revenue"].ne(0), df["Gross_Profit"] / df["Net_Revenue"], np.nan
)

# Save cleaned data for future reuse.
cleaned_path = OUTPUT_DIR / "cleaned_retail_sales.csv"
df.to_csv(cleaned_path, index=False)

# Post-cleaning validation.
remaining_missing = int(df.isna().sum().sum())
remaining_duplicates = int(df.duplicated().sum())
invalid_price_count = int((df["Unit_Price"].isna() | (df["Unit_Price"] <= 0)).sum())
invalid_cost_count = int((df["Unit_Cost"].isna() | (df["Unit_Cost"] <= 0)).sum())
revenue_errors = int((~np.isclose(df["Net_Revenue"], df["Gross_Revenue"] - df["Discount_Amount"], equal_nan=True)).sum())
profit_errors = int((~np.isclose(df["Gross_Profit"], df["Net_Revenue"] - df["Total_Cost"], equal_nan=True)).sum())

validation = pd.DataFrame([
    ("Rows before cleaning", len(raw_df)),
    ("Rows after cleaning", len(df)),
    ("Duplicate rows removed", duplicates_removed),
    ("Remaining exact duplicates", remaining_duplicates),
    ("Remaining missing cells", remaining_missing),
    ("Remaining invalid prices", invalid_price_count),
    ("Remaining invalid costs", invalid_cost_count),
    ("Net revenue calculation mismatches", revenue_errors),
    ("Gross profit calculation mismatches", profit_errors),
], columns=["Validation_Check", "Result"])
validation.to_csv(REPORT_DIR / "data_quality_validation.csv", index=False)
print("\nCleaning validation:")
print(validation.to_string(index=False))
print(f"\nCleaned dataset saved to: {cleaned_path}")

# -----------------------------------------------------------------------------
# 3. BUSINESS KPIs AND ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 3 — BUSINESS PERFORMANCE ANALYSIS")
print("=" * 72)

# Helper: weighted gross margin based on summed profit / summed revenue.
def weighted_margin(group: pd.DataFrame) -> float:
    revenue = group["Net_Revenue"].sum()
    if pd.isna(revenue) or np.isclose(revenue, 0):
        return np.nan
    return group["Gross_Profit"].sum() / revenue * 100

# Overall KPIs.
total_orders = df["Order_ID"].nunique(dropna=True)
total_quantity = df["Quantity"].sum()
gross_revenue = df["Gross_Revenue"].sum()
net_revenue = df["Net_Revenue"].sum()
gross_profit = df["Gross_Profit"].sum()
overall_margin = gross_profit / net_revenue * 100 if not np.isclose(net_revenue, 0) else np.nan
return_flag_normalized = df["Return_Flag"].astype("string").str.strip().str.casefold()
df["Is_Return"] = return_flag_normalized.eq("yes")
overall_return_rate = df["Is_Return"].mean() * 100

kpis = pd.DataFrame([
    ("Unique orders", total_orders),
    ("Net quantity (includes negative return quantities)", total_quantity),
    ("Gross revenue before discount", gross_revenue),
    ("Net revenue after discount", net_revenue),
    ("Estimated gross profit", gross_profit),
    ("Overall gross margin (%)", overall_margin),
    ("Return-flagged transaction rate (%)", overall_return_rate),
], columns=["KPI", "Value"])
kpis.to_csv(REPORT_DIR / "overall_kpis.csv", index=False)
print("Overall KPIs:")
print(kpis.to_string(index=False, formatters={"Value": lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"}))

# Standardized month, region, category, product and channel aggregations.
valid_dates = df.dropna(subset=["Order_Date"]).copy()
valid_dates["Month"] = valid_dates["Order_Date"].dt.to_period("M").astype(str)
monthly_sales = valid_dates.groupby("Month", as_index=True).agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum")
).sort_index()
regional_sales = df.groupby("Region").agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum"),
    Transactions=("Order_ID", "size"), Return_Rate_Pct=("Is_Return", "mean")
)
regional_sales["Return_Rate_Pct"] *= 100
regional_sales["Gross_Margin_Pct"] = regional_sales["Gross_Profit"] / regional_sales["Net_Revenue"].replace(0, np.nan) * 100
channel_metrics = df.groupby("Sales_Channel").agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum"), Transactions=("Order_ID", "size")
)
channel_metrics["Gross_Margin_Pct"] = channel_metrics["Gross_Profit"] / channel_metrics["Net_Revenue"].replace(0, np.nan) * 100
category_metrics = df.groupby("Product_Category").agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum"), Transactions=("Order_ID", "size"),
    Return_Rate_Pct=("Is_Return", "mean")
)
category_metrics["Gross_Margin_Pct"] = category_metrics["Gross_Profit"] / category_metrics["Net_Revenue"].replace(0, np.nan) * 100
category_metrics["Return_Rate_Pct"] *= 100
product_metrics = df.groupby("Product").agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum"), Transactions=("Order_ID", "size")
)
product_metrics["Gross_Margin_Pct"] = product_metrics["Gross_Profit"] / product_metrics["Net_Revenue"].replace(0, np.nan) * 100
customer_metrics = df.groupby("Customer_Name").agg(
    Net_Revenue=("Net_Revenue", "sum"), Gross_Profit=("Gross_Profit", "sum"), Transactions=("Order_ID", "size")
)

# Print key comparisons.
if not monthly_sales.empty:
    print(f"\nBest month by net revenue:  {monthly_sales['Net_Revenue'].idxmax()} ({monthly_sales['Net_Revenue'].max():,.2f})")
    print(f"Worst month by net revenue: {monthly_sales['Net_Revenue'].idxmin()} ({monthly_sales['Net_Revenue'].min():,.2f})")
if not regional_sales.empty:
    print(f"Best region by revenue: {regional_sales['Net_Revenue'].idxmax()} ({regional_sales['Net_Revenue'].max():,.2f})")
    print(f"Lowest-revenue region: {regional_sales['Net_Revenue'].idxmin()} ({regional_sales['Net_Revenue'].min():,.2f})")
    print(f"Highest return-rate region: {regional_sales['Return_Rate_Pct'].idxmax()} ({regional_sales['Return_Rate_Pct'].max():.2f}%)")
if not channel_metrics.empty:
    print(f"Highest-revenue channel: {channel_metrics['Net_Revenue'].idxmax()} ({channel_metrics['Net_Revenue'].max():,.2f})")
    print(f"Highest-margin channel: {channel_metrics['Gross_Margin_Pct'].idxmax()} ({channel_metrics['Gross_Margin_Pct'].max():.2f}%)")
if not category_metrics.empty:
    print(f"Highest-profit category: {category_metrics['Gross_Profit'].idxmax()} ({category_metrics['Gross_Profit'].max():,.2f})")
    print(f"Highest-margin category: {category_metrics['Gross_Margin_Pct'].idxmax()} ({category_metrics['Gross_Margin_Pct'].max():.2f}%)")
if not customer_metrics.empty and not np.isclose(net_revenue, 0):
    top10_customer_revenue = customer_metrics["Net_Revenue"].nlargest(10).sum()
    print(f"Top 10 customer revenue share: {top10_customer_revenue / net_revenue * 100:.2f}%")

# Save detailed analysis tables.
monthly_sales.to_csv(REPORT_DIR / "monthly_performance.csv")
regional_sales.sort_values("Net_Revenue", ascending=False).to_csv(REPORT_DIR / "regional_performance.csv")
channel_metrics.sort_values("Net_Revenue", ascending=False).to_csv(REPORT_DIR / "channel_performance.csv")
category_metrics.sort_values("Gross_Profit", ascending=False).to_csv(REPORT_DIR / "category_performance.csv")
product_metrics.sort_values("Gross_Profit", ascending=False).to_csv(REPORT_DIR / "product_performance.csv")
customer_metrics.sort_values("Net_Revenue", ascending=False).to_csv(REPORT_DIR / "customer_performance.csv")

print("\nTop 10 products by gross profit:")
print(product_metrics["Gross_Profit"].nlargest(10).to_string(float_format=lambda x: f"{x:,.2f}"))
print("\nTop 10 customers by net revenue:")
print(customer_metrics["Net_Revenue"].nlargest(10).to_string(float_format=lambda x: f"{x:,.2f}"))
print("\nRegional return rate (compare both rate and transaction count):")
print(regional_sales[["Return_Rate_Pct", "Transactions"]].sort_values("Return_Rate_Pct", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

# -----------------------------------------------------------------------------
# 4. VISUALIZATIONS — SAVE FIVE CHARTS
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 4 — VISUALIZATION AND BUSINESS STORYTELLING")
print("=" * 72)

# Chart 1: monthly net revenue.
if not monthly_sales.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_sales.index, monthly_sales["Net_Revenue"] / 1_000_000, marker="o")
    ax.set_title("Monthly Net Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net revenue (millions)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_monthly_revenue_trend.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Chart 2: gross profit by category.
category_profit = category_metrics["Gross_Profit"].sort_values()
if not category_profit.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    (category_profit / 1_000_000).plot(kind="barh", ax=ax)
    ax.set_title("Gross Profit by Product Category")
    ax.set_xlabel("Gross profit (millions)")
    ax.set_ylabel("Product category")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_profit_by_category.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Chart 3: top 10 products by gross profit.
top_products = product_metrics["Gross_Profit"].nlargest(10).sort_values()
if not top_products.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    (top_products / 1_000_000).plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Products by Gross Profit")
    ax.set_xlabel("Gross profit (millions)")
    ax.set_ylabel("Product")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_top_10_products_by_profit.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Chart 4: revenue vs gross profit by sales channel.
channel_chart = channel_metrics[["Net_Revenue", "Gross_Profit"]].sort_values("Net_Revenue", ascending=False)
if not channel_chart.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    (channel_chart / 1_000_000).plot(kind="bar", ax=ax)
    ax.set_title("Net Revenue vs. Gross Profit by Sales Channel")
    ax.set_xlabel("Sales channel")
    ax.set_ylabel("Amount (millions)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_channel_revenue_vs_profit.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Chart 5: return rate by region. Rate is shown alongside sample size in output table.
region_return_rates = regional_sales["Return_Rate_Pct"].sort_values()
if not region_return_rates.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    region_return_rates.plot(kind="barh", ax=ax)
    ax.set_title("Return-Flagged Transaction Rate by Region")
    ax.set_xlabel("Return-flagged transactions (%)")
    ax.set_ylabel("Region")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_return_rate_by_region.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# A compact findings file, based on consolidated/standardized labels.
findings = []
if not monthly_sales.empty:
    findings.extend([
        {"Metric": "Best month by net revenue", "Finding": str(monthly_sales["Net_Revenue"].idxmax())},
        {"Metric": "Worst month by net revenue", "Finding": str(monthly_sales["Net_Revenue"].idxmin())},
    ])
if not regional_sales.empty:
    findings.extend([
        {"Metric": "Highest-revenue region", "Finding": str(regional_sales["Net_Revenue"].idxmax())},
        {"Metric": "Lowest-revenue region", "Finding": str(regional_sales["Net_Revenue"].idxmin())},
        {"Metric": "Highest-return-rate region", "Finding": str(regional_sales["Return_Rate_Pct"].idxmax())},
        {"Metric": "Highest regional return rate (%)", "Finding": f"{regional_sales['Return_Rate_Pct'].max():.2f}%"},
    ])
if not category_metrics.empty:
    findings.extend([
        {"Metric": "Highest gross-profit category", "Finding": str(category_metrics["Gross_Profit"].idxmax())},
        {"Metric": "Highest gross-margin category", "Finding": str(category_metrics["Gross_Margin_Pct"].idxmax())},
    ])
if not channel_metrics.empty:
    findings.extend([
        {"Metric": "Highest-revenue sales channel", "Finding": str(channel_metrics["Net_Revenue"].idxmax())},
        {"Metric": "Highest-margin sales channel", "Finding": str(channel_metrics["Gross_Margin_Pct"].idxmax())},
    ])
findings.append({"Metric": "Overall gross margin (%)", "Finding": f"{overall_margin:.2f}%" if pd.notna(overall_margin) else "N/A"})
findings.append({"Metric": "Overall return-flagged transaction rate (%)", "Finding": f"{overall_return_rate:.2f}%"})
pd.DataFrame(findings).to_csv(REPORT_DIR / "key_findings.csv", index=False)

# Final output list.
print("\n" + "=" * 72)
print("PROJECT COMPLETE")
print("=" * 72)
print(f"Cleaned dataset: {cleaned_path}")
print(f"Charts folder:   {CHART_DIR}")
print(f"Reports folder:  {REPORT_DIR}")
print("\nImportant interpretation notes:")
print("- Negative quantities are retained and flagged as returns, as in the original cleaning decision.")
print("- Return rate is the share of rows flagged 'Yes'; it is not necessarily a unique-order return rate.")
print("- Revenue/profit/margin values are estimates because invalid prices/costs were median-imputed.")
print("- High return rates identify where to investigate; they do not prove the cause.")
