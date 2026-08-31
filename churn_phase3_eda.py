"""
Phase 3 - Exploratory Data Analysis (EDA)
Customer Churn Prediction (Telco Customer Churn dataset)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

CSV_PATH = "telco_churn_cleaned.csv"
df = pd.read_csv(CSV_PATH)

# Colors: churn is a binary status, not an arbitrary category, so we use
# fixed status colors rather than a generic categorical palette - green
# reads as "retained/good", red as "churned/critical" - and every plot
# uses the same two colors for the same two labels.
COLOR_NO = "#0ca30c"   # good / retained
COLOR_YES = "#d03b3b"  # critical / churned
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"

sns.set_style("white")
plt.rcParams.update({
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "font.family": "sans-serif",
})

# ---------------------------------------------------------------
# 1. Churn rate by contract type
# ---------------------------------------------------------------
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .reindex(["Month-to-month", "One year", "Two year"])
)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(contract_churn.index, contract_churn.values, color=COLOR_YES, width=0.5)
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn Rate by Contract Type")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for bar, val in zip(bars, contract_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
             ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
plt.tight_layout()
plt.savefig("plot1_churn_by_contract.png", dpi=150)
plt.show()

# Interpretation:
# Month-to-month customers churn at a dramatically higher rate than
# one- or two-year contract holders. This is the single strongest lever
# in the dataset: contract length acts as a retention mechanism by
# design (switching cost / commitment), so incentivizing month-to-month
# customers to move to annual plans - via discounts, bundled perks, or
# proactive outreach timed before renewal - is likely the highest-ROI
# retention action available.

# ---------------------------------------------------------------
# 2. Churn rate by tenure group
# ---------------------------------------------------------------
bins = [-1, 12, 24, 48, df["tenure"].max()]
labels = ["0-12", "13-24", "25-48", "49+"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

tenure_churn = (
    df.groupby("tenure_group", observed=True)["Churn"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .reindex(labels)
)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(tenure_churn.index, tenure_churn.values, color=COLOR_YES, width=0.5)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn Rate by Tenure Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for bar, val in zip(bars, tenure_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
             ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
plt.tight_layout()
plt.savefig("plot2_churn_by_tenure.png", dpi=150)
plt.show()

# Interpretation:
# Churn is heavily front-loaded: customers in their first year churn
# far more than long-tenured ones. This points to an onboarding /
# early-experience problem rather than a general dissatisfaction
# problem - customers who make it past roughly the first year become
# much more likely to stay. The business implication is to concentrate
# retention spend (proactive check-ins, onboarding support, early
# discounts) in the first 12 months rather than spreading it evenly
# across the whole customer base.

# ---------------------------------------------------------------
# 3. Correlation heatmap for numerical features
# ---------------------------------------------------------------
numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
numeric_df["Churn_numeric"] = (df["Churn"] == "Yes").astype(int)
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",   # diverging blue<->red, matches the palette's diverging pair
    vmin=-1, vmax=1,
    center=0,
    linewidths=1,
    linecolor="#fcfcfb",
    cbar_kws={"label": "Correlation"},
    ax=ax,
)
ax.set_title("Correlation Heatmap - Numerical Features")
plt.tight_layout()
plt.savefig("plot3_correlation_heatmap.png", dpi=150)
plt.show()

# Interpretation:
# TotalCharges is strongly positively correlated with tenure (longer
# customers naturally accumulate more billing), so the two carry
# overlapping information - worth flagging for feature selection later
# since including both may add multicollinearity without much extra
# signal. Churn correlates negatively with tenure and positively with
# MonthlyCharges, confirming numerically what the earlier bar charts
# showed visually: customers who are newer and pay more per month are
# the highest-risk segment.

# ---------------------------------------------------------------
# 4. Monthly charges distribution split by churn
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    data=df, x="Churn", y="MonthlyCharges", order=["No", "Yes"],
    hue="Churn", hue_order=["No", "Yes"], legend=False,
    palette={"No": COLOR_NO, "Yes": COLOR_YES}, ax=ax, width=0.5,
)
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges ($)")
ax.set_title("Monthly Charges by Churn Status")
ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("plot4_monthly_charges_by_churn.png", dpi=150)
plt.show()

# Interpretation:
# Churned customers have a visibly higher median MonthlyCharges than
# retained customers, and their charges are more tightly clustered at
# the high end. This suggests price sensitivity is a real driver of
# churn - customers on expensive plans (likely with add-ons like
# streaming or premium tech support) may not perceive enough value for
# the cost, making them good candidates for retention offers such as
# discounts or a review of their plan/add-on mix before they leave.
