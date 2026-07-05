# Churn_Analysis_BCG

# PowerCo SME Customer Churn Prediction

End-to-end data science project predicting customer churn for PowerCo, a major gas and electricity utility supplying small and medium-sized enterprises (SMEs). Completed as part of the **BCG X Data Science Job Simulation** (Forage).

## Business Problem

PowerCo has been losing SME customers since the liberalization of the European energy market, and the leading internal hypothesis was that churn is driven by **price sensitivity** — customers leaving because their prices went up. The goals of this project were to:

1. Test whether price sensitivity is actually a significant driver of churn
2. Build a predictive model that identifies which customers are most likely to churn in the next 3 months
3. Recommend a data-driven retention strategy

## Data

Two datasets covering ~14,600 SME clients:

| File | Description |
|---|---|
| `client_data.csv` | One row per client: electricity/gas consumption, contract dates, forecasts, margins, subscribed power, sales channel, and the churn label (churned within 3 months) |
| `price_data.csv` | Monthly price records per client for one year: variable (energy) and fixed (power) prices across off-peak, peak, and mid-peak periods |

Some categorical fields are hashed to preserve privacy while retaining predictive value.

## Project Workflow

### 1. Exploratory Data Analysis
- Data quality checks: missing values, duplicates, ID consistency across files
- Target analysis: **~9.7% churn rate** (imbalanced classification problem)
- Distribution analysis revealing heavy right skew in consumption and margin variables
- Price trends over time and churn relationships across categorical/numerical features

### 2. Hypothesis Testing — Is Churn Driven by Price Sensitivity?
- **Mann-Whitney U tests**: 6 of 7 price features differ significantly between churned and retained clients
- **Chi-square test** on binary "price increased" flag: not significant (p ≈ 0.10) — churn responds to the *magnitude* of price changes, not the mere presence of an increase
- **Logistic regression with controls**: price effects persist but are small after controlling for tenure, margin, and consumption
- **Conclusion**: price sensitivity is real but *weak* — a secondary driver behind contract lifecycle and client value characteristics

### 3. Feature Engineering
Lean, hypothesis-driven feature set (no "just in case" columns):
- **Price features**: December-vs-January off-peak price difference (the core case-study feature), mean off-peak price levels
- **Date features**: client tenure, months to renewal, months since last product modification
- **Transformations**: `log1p` on heavily skewed consumption/power columns
- **Encoding**: rare-category grouping + one-hot for sales channel and campaign origin; boolean conversion for gas subscription

### 4. Modeling — Random Forest
- Stratified train/validation/test split (feature selection performed strictly on training data — no leakage)
- **Permutation importance pruning**: features with zero or negative importance on held-out data removed as noise
- Class imbalance handled with `class_weight='balanced'`
- Hyperparameter tuning via 5-fold stratified `GridSearchCV` optimizing ROC-AUC
- Decision-threshold analysis to align the model with retention-campaign economics

### 5. Results

| Metric | Value |
|---|---|
| Test ROC-AUC | **0.72** |
| Test PR-AUC | 0.31 (vs. 0.10 random baseline — ~3x lift) |
| Top churn drivers | Client tenure, renewal timing, margins, consumption patterns |
| Price features | Statistically significant but mid-ranked in importance |

### 6. Key Business Recommendations
- **Don't discount everyone** — price is a secondary driver, so a blanket price cut sacrifices margin on ~90% of clients who were never going to leave
- **Target instead**: focus a retention offer (e.g., 20% discount) on the highest-risk decile flagged by the model, reaching ~2–3x more actual churners per euro spent
- **Time outreach to renewal windows**, where churn risk peaks
- **Pilot with a control group** to measure true uplift before scaling

## Repository Structure

```
├── data/
│   ├── client_data.csv
│   └── price_data.csv
├── notebooks/
│   ├── 01_EDA_churn_dataset.ipynb           # Exploratory data analysis
│   ├── 02_Feature_Engineering_churn.ipynb   # Feature creation & dataset prep
│   ├── 03_RandomForest_churn_model.ipynb    # Model training & evaluation
│   └── BCG_Job_Simulation_rewritten.ipynb   # End-to-end lean pipeline
├── outputs/
│   ├── data_for_predictions.csv             # Model-ready dataset
│   ├── churn_rf_final.joblib                # Trained model + feature list
│   └── Churn_Executive_Summary.pdf          # One-slide stakeholder summary
└── README.md
```

## Tech Stack

- **Python 3** — pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Modeling** — scikit-learn (RandomForestClassifier, GridSearchCV, permutation_importance)
- **Statistics** — SciPy (Mann-Whitney U, chi-square), statsmodels (logistic regression)

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels joblib

# Run notebooks in order (place the CSVs in the working directory)
jupyter notebook notebooks/01_EDA_churn_dataset.ipynb
```

Each notebook is self-contained and runs top to bottom. The feature engineering notebook exports `data_for_predictions.csv`, which the modeling notebook consumes.

## Key Learnings

- With imbalanced targets, **accuracy is meaningless** — stratified splits, class weighting, and PR-AUC/recall are what matter
- A Random Forest's near-perfect training score is expected, not necessarily overfitting; **compare CV score to test score** instead
- **Permutation importance on held-out data** beats built-in Gini importance for identifying genuinely useful features
- Statistical significance with n ≈ 14,000 is cheap; **effect sizes carry the business story**
- The most valuable insight contradicted the client's hypothesis: churn is a lifecycle problem more than a pricing problem

---

*This project was completed as part of the BCG X Data Science Job Simulation on Forage. The data is simulated and provided for educational purposes.*
