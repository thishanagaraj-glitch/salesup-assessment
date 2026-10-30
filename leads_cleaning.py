# %% [markdown]
# # SalesUp assessment: leads.csv cleaning and monthly summary
#
# Steps: load, standardize `source` (and `status`), remove blank and duplicate rows,
# fix dates and add a `month` column, clean `deal_value`, flag outliers (IQR),
# and build the monthly summary (total leads, won leads, conversion rate, total deal value).

# %%
import re
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "leads.csv"

# Duplicates: exact repeats of a lead_id are always removed.
# If True, leads that share the same email (different lead_id) are also collapsed,
# keeping the earliest created_date (same rule used in the SQL part).
DEDUP_BY_EMAIL = True

# %% [markdown]
# ## 1. Load the raw data
# `utf-8-sig` strips the BOM that otherwise turns the first column into `﻿lead_id`.
# Everything is read as text so no value is silently converted.

# %%
raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig", keep_default_na=False)
raw.columns = raw.columns.str.strip()
print("raw shape:", raw.shape)
raw.head(10)

# %% [markdown]
# ## 2. Remove blank rows

# %%
df = raw.apply(lambda col: col.str.strip())
blank_mask = (df == "").all(axis=1)
n_blank = int(blank_mask.sum())
df = df.loc[~blank_mask].copy()
print(f"blank rows removed: {n_blank}  -> {len(df)} rows left")

# %% [markdown]
# ## 3. Standardize `source` and `status`
# Spelling, case and stray spaces differ ("Linkedin", " LINKEDIN", "Web site", "cold call ").
# Strip spaces, lowercase, drop inner spaces, then map to one canonical label.

# %%
SOURCE_MAP = {
    "linkedin": "LinkedIn",
    "website": "Website",
    "coldcall": "Cold Call",
    "referral": "Referral",
    "indeed": "Indeed",
}


def std_source(value: str) -> str:
    key = re.sub(r"\s+", "", str(value).lower())
    return SOURCE_MAP.get(key, "UNKNOWN")


df["source_raw"] = df["source"]
df["source"] = df["source"].map(std_source)
df["status"] = df["status"].str.strip().str.title()  # won / WON / 'lost ' -> Won / Lost

print("raw distinct sources     :", df["source_raw"].nunique())
print("standardized sources     :", sorted(df["source"].unique()))
print("standardized status      :", sorted(df["status"].unique()))
assert "UNKNOWN" not in set(df["source"]), "unmapped source value found"
df["source"].value_counts()

# %% [markdown]
# ## 4. Remove duplicates
# * Step A: exact repeats of the same lead (same `lead_id`), keep the first.
# * Step B (switch `DEDUP_BY_EMAIL`): same email on different leads, keep the earliest `created_date`.
#   Blank emails are never treated as duplicates of each other.

# %%
df["lead_id"] = df["lead_id"].astype(int)
df["rep_id"] = df["rep_id"].astype(int)

before = len(df)
dup_ids = df.loc[df.duplicated("lead_id", keep="first"), "lead_id"].tolist()
df = df.drop_duplicates("lead_id", keep="first").copy()
print(f"duplicate lead_id rows removed: {before - len(df)}  ids={dup_ids}")

# %% [markdown]
# ## 5. Fix dates and add the `month` column
# Four formats are mixed: `2026-09-06`, `16.09.2026`, `09/10/2026` (day first) and `25-Sep-2026`.
# Slash dates are day/month: lead 5 reads `10/09/2026` and the SQL table has 2026-09-10 for it.
# Each format is parsed explicitly so nothing is guessed.

# %%


def parse_date(text: str) -> pd.Timestamp:
    text = str(text).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    return pd.NaT


df["created_date"] = df["created_date"].map(parse_date)
assert df["created_date"].notna().all(), "unparsed dates: " + str(df.loc[df["created_date"].isna()])
df["month"] = df["created_date"].dt.strftime("%Y-%m")
print("date range:", df["created_date"].min().date(), "to", df["created_date"].max().date())
df[["lead_id", "created_date", "month"]].head()

# %% [markdown]
# ## 6. Clean `deal_value`
# Remove the rupee sign and thousands separators. Blank stays missing (NaN), it is not turned into 0.

# %%


def parse_money(text: str) -> float:
    cleaned = re.sub(r"[₹,\s]", "", str(text))
    return float(cleaned) if cleaned not in ("", "-") else np.nan


df["deal_value"] = df["deal_value"].map(parse_money)

# %% [markdown]
# ## 7. Email duplicates (optional step B)

# %%
df["email"] = df["email"].str.lower()
has_email = df["email"] != ""
email_dups = df[has_email & df.duplicated("email", keep=False)].sort_values(["email", "created_date"])
print("leads sharing an email with another lead:")
print(email_dups[["lead_id", "email", "created_date", "status"]].to_string(index=False))

if DEDUP_BY_EMAIL:
    ordered = df.sort_values(["created_date", "lead_id"])
    drop_idx = ordered[has_email.reindex(ordered.index) & ordered.duplicated("email", keep="first")].index
    print("removed (later duplicates):", df.loc[drop_idx, "lead_id"].tolist())
    df = df.drop(index=drop_idx)

df = df.sort_values("lead_id").reset_index(drop=True)
print("clean rows:", len(df))

# %% [markdown]
# ## 8. Data-quality flags and outlier flagging (IQR)
# * `is_won` marks status Won.
# * Outliers use the 1.5 x IQR rule on **positive Won deal values**.
# * `negative_value`, `won_missing_value` and `won_zero_value` catch values that are wrong without being "large".
# * Rows with status other than Won should carry no deal value; those are flagged too.

# %%
df["is_won"] = df["status"].eq("Won")

won_vals = df.loc[df["is_won"] & (df["deal_value"] > 0), "deal_value"]
q1, q3 = won_vals.quantile(0.25), won_vals.quantile(0.75)
iqr = q3 - q1
low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Won deal values: Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
print(f"IQR fences: low={low_fence:,.0f}  high={high_fence:,.0f}")

df["negative_value"] = df["deal_value"] < 0
df["won_missing_value"] = df["is_won"] & df["deal_value"].isna()
df["won_zero_value"] = df["is_won"] & df["deal_value"].eq(0)
df["value_on_non_won"] = ~df["is_won"] & df["deal_value"].fillna(0).ne(0)
df["iqr_outlier"] = df["is_won"] & (df["deal_value"] > high_fence)
df["outlier_flag"] = (
    df["iqr_outlier"] | df["negative_value"] | df["value_on_non_won"]
)

flag_cols = ["iqr_outlier", "negative_value", "won_missing_value", "won_zero_value", "value_on_non_won"]
print(df[flag_cols].sum().rename("rows flagged"))
df.loc[df[flag_cols].any(axis=1), ["lead_id", "source", "status", "month", "deal_value"] + flag_cols]

# %% [markdown]
# ## 9. Monthly summary
# * **total_leads**: all clean leads created in the month
# * **won_leads**: leads with status Won
# * **conversion_rate_pct**: won_leads / total_leads x 100
# * **total_deal_value**: sum of Won deal values after removing invalid values (negatives and values on non-Won rows are excluded; missing values count as 0)
# * **total_deal_value_ex_outliers**: the same, also excluding IQR outliers

# %%
valid_won = df["is_won"] & ~df["negative_value"]
df["deal_value_valid"] = df["deal_value"].where(valid_won, 0).fillna(0)
df["deal_value_ex_outliers"] = df["deal_value_valid"].where(~df["iqr_outlier"], 0)

monthly = (
    df.groupby("month")
    .agg(
        total_leads=("lead_id", "count"),
        won_leads=("is_won", "sum"),
        total_deal_value=("deal_value_valid", "sum"),
        total_deal_value_ex_outliers=("deal_value_ex_outliers", "sum"),
    )
    .reset_index()
)
monthly["won_leads"] = monthly["won_leads"].astype(int)
monthly["conversion_rate_pct"] = (100 * monthly["won_leads"] / monthly["total_leads"]).round(2)
monthly = monthly[
    ["month", "total_leads", "won_leads", "conversion_rate_pct", "total_deal_value", "total_deal_value_ex_outliers"]
]

total_row = {
    "month": "TOTAL",
    "total_leads": monthly["total_leads"].sum(),
    "won_leads": monthly["won_leads"].sum(),
    "conversion_rate_pct": round(100 * monthly["won_leads"].sum() / monthly["total_leads"].sum(), 2),
    "total_deal_value": monthly["total_deal_value"].sum(),
    "total_deal_value_ex_outliers": monthly["total_deal_value_ex_outliers"].sum(),
}
monthly_out = pd.concat([monthly, pd.DataFrame([total_row])], ignore_index=True)
print(monthly_out.to_string(index=False, formatters={
    "total_deal_value": "{:,.0f}".format,
    "total_deal_value_ex_outliers": "{:,.0f}".format,
}))
monthly_out

# %% [markdown]
# ## 10. Save the outputs

# %%
clean_cols = [
    "lead_id", "rep_id", "source", "status", "created_date", "month", "deal_value", "email",
    "outlier_flag", "iqr_outlier", "negative_value", "won_missing_value", "won_zero_value", "value_on_non_won",
]
out = df[clean_cols].copy()
out["created_date"] = out["created_date"].dt.strftime("%Y-%m-%d")
out.to_csv("leads_clean.csv", index=False)
monthly_out.to_csv("monthly_summary.csv", index=False)
print("saved leads_clean.csv and monthly_summary.csv")
