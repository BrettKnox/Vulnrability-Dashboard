# Predicting Poor Mental Health at the City Council District Level

> **CAP 4922 — Data Science Capstone Project**
> Team K² · Brett Knox & Connor Kurrack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![D3.js](https://img.shields.io/badge/D3.js-Dashboard-F9A03C?style=flat-square&logo=d3dotjs&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-00d4aa?style=flat-square)

---

##  Overview

This project examines how **district-level socioeconomic, environmental, and healthcare access conditions** predict poor adult mental health across city council districts in 14 major U.S. cities. The target variable is the percentage of adults reporting poor mental health — defined as poor mental health for **≥ 14 days** in the past month (CDC PLACES, 2023).

The project is split into two analytical branches:

| Branch | Focus | Lead |
|--------|-------|------|
| **Regression + Composite Index** | Feature engineering, Ridge/Lasso/RF regression, PCA-weighted risk index | Brett Knox |
| **Clustering + District Profiling** | K-means cluster analysis, district pattern identification | Connor Kurrack |

Both branches share a common cleaned dataset and converge into a district-level **interactive dashboard** for nonprofit and community-facing use.

---

##  Data

### Coverage

**239 city council districts** across **14 U.S. cities** (cross-sectional, 2020–2025):

`Austin` · `Charlotte` · `Columbus` · `Denver` · `Indianapolis` · `Jacksonville` · `Louisville` · `Miami` · `Nashville` · `Orlando` · `Philadelphia` · `San Antonio` · `San Francisco` · `Tampa`

### Source

| Field | Detail |
|-------|--------|
| **Provider** | Mysidewalk (district-level public indicators) |
| **Raw dimensions** | 239 rows × 27 columns |
| **Unit of analysis** | Single city council district |
| **Time range** | 2020–2025 (cross-sectional) |

### Feature Groups

<details>
<summary><strong>View all feature groups</strong></summary>

**Housing / Economic Stress**
- Excess Housing Costs (≥30% of income)
- People Below Poverty Level
- People with / without Health Insurance

**Household Composition**
- Family Household with Married Couple
- Family Household with Single Male Householder
- Family Household with Single Female Householder

**Education**
- Less than 9th Grade · 9th–12th No Diploma · High School Degree
- Some College · Associate · Bachelor's · Graduate Degree

**Environmental Justice**
- Lead Paint Environmental Justice Index
- Drinking Water Non-Compliance
- Drinking Water Non-Compliance EJI

**Healthcare Access**
- Mental Health Providers (Male / Female / Total)
- Total Healthcare Workers
- Mental Health Provider Ratio

**Target Variable**
- `Poor Mental Health Among Adults (2023)` — % reporting ≥14 poor mental health days/month

</details>

---

##  Methodology

### 1. Data Cleaning & Preprocessing

- **Imputation:** Median imputation for `MH Provider Ratio` (only feature with missing values; 4/239 rows, 1.67%)
- **Outlier treatment:** IQR-based winsorization at 1st/99th percentile for all final features
- **Standardization:** Z-score normalization applied before modeling

### 2. Feature Engineering

Raw count variables were converted to **population-normalized rates** using a population proxy (`Insured + Uninsured`), which transformed signal quality dramatically:

```
pop_proxy         = Insured + Uninsured
poverty_rate      = Below Poverty / pop_proxy
housing_cost_rate = Excess Housing Costs / pop_proxy
grad_edu_rate     = Graduate Degrees / pop_proxy
bach_edu_rate     = Bachelor's Degrees / pop_proxy
uninsured_rate    = Uninsured / pop_proxy
...
```

Log transformations were applied to right-skewed environmental features (`Lead Paint EJI`, `Drinking Water EJI`, `MH Provider Ratio`).

### 3. Feature Selection (VIF + Correlation)

**11 final features** selected via correlation screening and Variance Inflation Factor analysis (VIF < 10 threshold):

| Feature | r with Target | VIF |
|---------|:-------------:|:---:|
| `poverty_rate` | +0.707 | 5.37 |
| `married_hh_rate` | −0.667 | 5.48 |
| `grad_edu_rate` | −0.657 | 5.98 |
| `single_female_hh_rate` | +0.637 | 3.78 |
| `bach_edu_rate` | −0.632 | 7.11 |
| `uninsured_rate` | +0.488 | 2.68 |
| `Lead Paint EJI` | +0.436 | 4.10 |
| `housing_cost_rate` | +0.276 | 3.38 |
| `low_edu_rate` | +0.178 | 2.37 |
| `log_Drinking Water EJI` | +0.104 | 3.36 |
| `log_MH Provider Ratio` | +0.086 | 1.52 |

### 4. Modeling

Four model families evaluated via **5-fold cross-validation**:

| Model | CV R² | CV RMSE | Notes |
|-------|:-----:|:-------:|-------|
| **Ridge + City Fixed Effects** | **0.9028** | **0.751** | Best overall; city dummies capture local baseline |
| Ridge (no city) | 0.6978 | 1.340 | α = 8.11 |
| Lasso (no city) | 0.6972 | 1.340 | α = 0.015; zeroed `housing_cost_rate` |
| Random Forest | 0.6250 | 1.484 | High train R² (0.874) → some overfitting |
| Gradient Boosting | 0.6261 | 1.481 | Severe overfit (train R² = 0.995) |

**OLS with City Fixed Effects** (statsmodels) confirmed the regression structure:
- R² = 0.927, Adj. R² = 0.918, F-stat = 112.4 (p < 0.001)
- Significant predictors: `married_hh_rate`, `poverty_rate`, `grad_edu_rate`, `bach_edu_rate`, `single_female_hh_rate`, `low_edu_rate`

### 5. Leave-One-City-Out Cross-Validation (LOCO-CV)

Standard k-fold inflates performance estimates by mixing districts across cities. LOCO-CV answers: *"If we only have data from 13 cities, how well can we predict the 14th?"*

| Held-Out City | Ridge R² | Ridge MAE |
|---------------|:--------:|:---------:|
| Austin | 0.916 | 0.444 |
| Jacksonville | 0.821 | 0.560 |
| Philadelphia | 0.766 | 0.698 |
| Charlotte | 0.556 | 1.089 |
| Miami | −2.421 | 3.288 |
| Nashville | −0.477 | 2.722 |
| **Mean** | **0.065** | **1.304 %pts** |

> Cities with negative LOCO R² (Miami, Nashville, Denver, San Antonio) exhibit unique local patterns not explained by the shared feature set — suggesting unobserved city-specific factors or data idiosyncrasies.

### 6. Composite Risk Index (PCA)

A **0–100 composite index** was constructed using PCA-weighted principal components:

- 4 PCs required to explain ≥80% variance (PC1: 44.9%, PC2: 19.5%, PC3: 9.7%, PC4: 8.3%)
- Weighted by each PC's absolute correlation with the target
- Final index correlation with target: **r = 0.688** (p < 0.001)

Top contributing features to the index:

| Feature | Weight | Direction |
|---------|:------:|:---------:|
| `married_hh_rate` | 0.290 | ↓ risk |
| `poverty_rate` | 0.286 | ↑ risk |
| `uninsured_rate` | 0.265 | ↑ risk |
| `low_edu_rate` | 0.260 | ↑ risk |
| `Lead Paint EJI` | 0.240 | ↑ risk |

## 7. Clustering (K-Means)

![Method](https://img.shields.io/badge/Method-K--Means-blue)
![Type](https://img.shields.io/badge/Learning-Unsupervised-orange)
![Primary_Model](https://img.shields.io/badge/Primary_Model-k%3D4-success)
![Metric](https://img.shields.io/badge/Metric-Silhouette_Score-lightgrey)

K-Means clustering was applied to Z-score normalized features to identify recurring district profiles defined by shared structural conditions. The objective was to segment districts based on underlying socioeconomic and environmental characteristics rather than population size.

---

### 📊 Model Selection

- Evaluated **k ∈ [2, 8]** using:
  - Elbow Method (inertia)
  - Silhouette Score (primary metric)

| k | Silhouette Score | Interpretation |
|--|------------------|----------------|
| 2 | **~0.31** | Strong separation (binary split) |
| 4 | ~0.20 | Moderate separation, higher interpretability |

- **k = 2** produced the strongest statistical separation  
- **k = 4** was selected as the primary model for its ability to capture **meaningful subgroups**

> ⚖️ **Trade-off:** Slightly lower separation, but significantly improved interpretability and real-world applicability

---

### 🧠 Cluster Structure (k = 4)

![Clusters](https://img.shields.io/badge/Clusters-4-blueviolet)

The four-cluster solution reveals a **continuous socioeconomic gradient**:

| Cluster | Profile | Mental Health Risk |
|--------|--------|--------------------|
| 🔴 0 | Extreme disadvantage | Highest (~20%) |
| 🟠 3 | Moderate-high risk | Elevated |
| 🔵 2 | Transitional | Moderate |
| 🟢 1 | Advantaged | Lowest (~16%) |

> Districts are not cleanly separated — they exist along a **structured continuum of risk**

---

### 🔍 Key Drivers of Clustering

![Drivers](https://img.shields.io/badge/Drivers-Socioeconomic-critical)

Clusters are primarily defined by:

- **Household Structure**
  - single-parent household share  
  - single-female household share  

- **Socioeconomic Status**
  - poverty rate  
  - educational attainment  

- **Access & Stability**
  - insurance coverage  
  - housing cost burden  

- **Environmental Exposure**
  - environmental risk index  

---

### 🔥 Key Insight

> Mental health disparities are driven primarily by **structural socioeconomic conditions**, not healthcare availability.

- Healthcare variables showed **minimal separation across clusters**
- Socioeconomic variables consistently dominated clustering behavior

---

### ⚠️ DBSCAN Comparison

![Alt Model](https://img.shields.io/badge/Comparison-DBSCAN-lightgrey)

DBSCAN was tested to identify density-based clusters and outliers.

**Results:**
- 1 dominant cluster + noise  
- High sensitivity to ε  
- No stable cluster structure  

**Conclusion:**
> Data does not contain strong density-based groupings — instead follows a **continuous gradient**

---

### 📍 Localized Insight: Jacksonville Case Study

![Case Study](https://img.shields.io/badge/Case_Study-Jacksonville_FL-blue)

Cluster assignments reveal **significant intra-city variation**:

- Districts span **all 4 clusters**
- Mental health rates range from **~16% to >21%**
- High-risk and low-risk districts exist within the same city

**Example:**
- 🔴 Districts 9 & 10 → highest risk  
- 🟢 Districts 2, 3, 6 → lower risk  

> 📌 **Key takeaway:** City-level averages obscure important local disparities

---

### ✅ Summary

![Status](https://img.shields.io/badge/Status-Validated-success)

- Clustering reveals a **clear socioeconomic gradient**
- **k = 4** provides actionable segmentation into risk tiers
- Results are:
  - interpretable  
  - consistent across features  
  - aligned with real-world conditions  

---


##  Key Findings

- **Rate normalization was essential.** Raw counts had near-zero correlations with the target; normalized rates revealed correlations up to r = 0.71.
- **Poverty rate is the single strongest predictor** (r = +0.71), consistent with economic insecurity as a documented driver of psychological distress.
- **Higher education is strongly protective** — `grad_edu_rate` (r = −0.66) and `bach_edu_rate` (r = −0.63) are among the top inverse predictors.
- **Lead Paint EJI** (r = +0.44 log-transformed) is the strongest environmental predictor, confirming environmental burden as a compounding stressor.
- **City fixed effects explain substantial variance** — jumping from R² ≈ 0.70 to R² ≈ 0.93 — indicating that unmeasured city-level factors matter as much as individual predictors.
- **LOCO-CV reveals generalization limits.** The model transfers well to most cities but struggles with cities that have unusual structural profiles (Miami, Nashville).

### Highest-Risk Districts (Composite Index)

| Rank | City | Index | Poor MH % |
|------|------|:-----:|:---------:|
| 1 | San Antonio | 100.0 | 19.1% |
| 2 | Miami | 99.3 | 15.3% |
| 3 | Miami | 98.4 | 15.7% |
| 5 | Philadelphia | 91.4 | 22.0% |
| 7 | Jacksonville | 87.1 | 21.6% |

---

##  Interactive Dashboard

An HTML/D3.js dashboard is included for visualizing the composite index and cluster assignments across counties.

**Features:**
- US county choropleth colored by composite index (0–100 heat scale)
- Toggle between Index, Cluster, and Poor MH % coloring
- Upload your own CSV to map custom composite scores and cluster labels to counties
- City filter, risk range slider, district-level rankings table
- Hover tooltips and click-to-pin for county inspection
- Export current view as CSV

**To use:** Open `dashboard/mental-health-risk-dashboard.html` in any modern browser. No server required.

**CSV Upload Format:**

```
fips, composite_index, cluster, city, poor_mh_pct, district_id
48453, 72.4, 3, Austin, 18.2, 109
...
```

> If no FIPS code is provided, the app will auto-map using the city name for the 14 study cities.

---

##  Repository Structure

```
├── data/
│   ├── MasterTable.xlsx              # Raw merged dataset (239 × 27)
│   ├── modeling_dataset_final.csv    # Cleaned + engineered features
│   ├── district_risk_rankings.csv    # Composite index per district
│   ├── loco_cv_results.csv           # LOCO-CV output by city
│   ├── model_comparison_results.csv  # CV metrics for all models
│   ├── lasso_coefficients.csv        # Lasso feature weights
│   ├── ridge_coefficients.csv        # Ridge feature weights
│   └── index_feature_contributions.csv
│
├── notebooks/
│   ├── Deliverable-3-EDA.ipynb       # Exploratory data analysis
│   └── Modeling.ipynb                # Feature engineering + all models
│
├── dashboard/
│   └── mental-health-risk-dashboard.html  # Interactive D3 map
│
├── deliverables/
│   └── Deliverable-3.docx            # EDA report
│
└── README.md
```

---

##  Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

### Run the Modeling Notebook

```bash
jupyter notebook notebooks/Modeling.ipynb
```

### Open the Dashboard

```bash
open dashboard/mental-health-risk-dashboard.html
# or just double-click the file in your file explorer
```

---

##  Research Hypotheses

| # | Hypothesis |
|---|-----------|
| H1 | Higher `poverty_rate` → higher poor mental health (economic insecurity drives psychological distress) |
| H2 | Higher `grad_edu_rate` → lower poor mental health (education provides protective social/economic resources) |
| H3 | Higher `Lead Paint EJI` → higher poor mental health (environmental burden compounds cumulative stressor load) |
| H4 | Higher `single_female_hh_rate` → higher poor mental health (single-parent households face elevated economic and social strain) |
| H5 | Ridge + city fixed effects outperforms naive OLS (geographic clustering requires structured control) |

---

##  Data Sources

| Source | Variables | Years |
|--------|-----------|-------|
| U.S. Census Bureau (ACS) | Education, poverty, household composition, insurance | 2020–2024 |
| CDC PLACES | Poor mental health % among adults | 2023 |
| CDC EJI (Environmental Justice Index) | Lead Paint EJI, Drinking Water EJI | 2024 |
| HRSA / Mysidewalk | Mental health provider counts and ratio | 2025 |

---

##  Limitations

- **Ecological fallacy:** All findings are at the district level. Community-level correlations do not imply individual-level relationships.
- **Cross-sectional design:** No causal inference can be drawn from this observational data.
- **City-specific unobservables:** LOCO-CV shows that several cities (Miami, Nashville) have patterns driven by factors not captured in the current feature set.
- **Population proxy:** Insured + Uninsured is used as a population estimate; it may undercount undocumented residents or those not captured in ACS estimates.
- **Environmental sparsity:** Drinking Water Non-Compliance is zero for 61.5% of districts, limiting its modeling utility.

---

##  Team

| Name | Role |
|------|------|
| **Brett Knox** | Regression analysis, feature engineering, composite index, Ridge/Lasso/RF modeling, dashboard |
| **Connor Kurrack** | EDA, clustering analysis, K-means district profiling, data quality audit |

**Course:** CAP 4922 — Data Science Capstone · University of North Florida
