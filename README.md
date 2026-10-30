# SalesUp Assessment: Part B (Python / pandas)

Cleans the messy `leads.csv` and builds a monthly summary.

**What the notebook does**
- Standardizes the `source` column (18 spellings reduced to 5)
- Removes blank rows and duplicate leads
- Fixes mixed date formats and adds a `month` column
- Builds the monthly summary: total leads, won leads, conversion rate, total deal value
- Flags outliers using the 1.5 x IQR rule

**Files**
- `leads_cleaning.ipynb`: notebook with outputs
- `leads_cleaning.py`: same code as a script
- `leads.csv`: raw input
- `monthly_summary.csv`: output table
