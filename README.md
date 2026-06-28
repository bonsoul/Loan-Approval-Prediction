#  Loan Approval Prediction

> Logistic regression model predicting credit decisions with **92% accuracy** — built on segmentation, correlation analysis, and CIBIL scoring.

---

## Analysis Pipeline & Findings

**1. Applicant Segmentation**
Applicants were profiled across four dimensions — income band (Low / Lower Middle / Upper Middle / High), number of dependents (Low / Moderate / High), education (Graduate / Not Graduate), and employment type (salaried / self-employed) — to surface structural differences in the applicant pool.

**2. Approval Rate Analysis**

| Factor | Finding |
|---|---|
| Education | Graduates had a higher loan approval rate than non-graduates |
| Employment | Salaried applicants were approved at a higher rate than the self-employed |
| Dependents | Chi-square test confirmed a **significant relationship** between number of dependents and loan status (p < 0.05) |
| CIBIL Score | Approval rates increased sharply with rating band — Excellent-rated applicants approved at the highest rate, Poor-rated at the lowest |

**3. Correlation Analysis**

| Test | Variables | Result |
|---|---|---|
| ANOVA | Income vs. all asset types | **Significant** — asset values and income are meaningfully related (p < 0.05) |
| T-Test | Asset values (residential, commercial, luxury, bank) vs. loan status | All four asset types showed **statistically significant** differences between approved and rejected applicants |
| T-Test | Income vs. loan amount | **Significant** relationship confirmed (p < 0.05) |
| Pearson | CIBIL score vs. loan status | Strong positive correlation — higher CIBIL score strongly predicts approval |
| Chi-Square | Self-employment vs. loan status | Significant association; self-employed applicants face lower approval odds |

**4. Predictive Model**

Logistic regression trained on numeric features (CIBIL score, income, assets, loan amount, loan term, dependents) with 70/30 train-test split and standard scaling.

| Set | Accuracy |
|---|---|
| Training | ~92% |
| Test | ~92% |

---

## Stack

`Python` · `Pandas` · `Scikit-learn` · `Matplotlib` · `Seaborn`
