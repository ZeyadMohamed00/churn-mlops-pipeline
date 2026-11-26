# EDA: IBM Telco Customer Churn

- Rows: 7043, columns: 20 features + target
- Churn rate: 26.5% (imbalanced: accuracy alone is misleading)
- Missing: TotalCharges blank for 11 rows with tenure 0 (set to 0)

## Numeric summary

|       |   tenure |   MonthlyCharges |   TotalCharges |   SeniorCitizen |
|:------|---------:|-----------------:|---------------:|----------------:|
| count |  7043    |          7043    |        7043    |         7043    |
| mean  |    32.37 |            64.76 |        2279.73 |            0.16 |
| std   |    24.56 |            30.09 |        2266.79 |            0.37 |
| min   |     0    |            18.25 |           0    |            0    |
| 25%   |     9    |            35.5  |         398.55 |            0    |
| 50%   |    29    |            70.35 |        1394.55 |            0    |
| 75%   |    55    |            89.85 |        3786.6  |            0    |
| max   |    72    |           118.75 |        8684.8  |            1    |

## Churn rate by category

### Contract

| Contract       |   churn_rate |    n |
|:---------------|-------------:|-----:|
| Month-to-month |        0.427 | 3875 |
| One year       |        0.113 | 1473 |
| Two year       |        0.028 | 1695 |

### InternetService

| InternetService   |   churn_rate |    n |
|:------------------|-------------:|-----:|
| DSL               |        0.19  | 2421 |
| Fiber optic       |        0.419 | 3096 |
| No                |        0.074 | 1526 |

### PaymentMethod

| PaymentMethod             |   churn_rate |    n |
|:--------------------------|-------------:|-----:|
| Bank transfer (automatic) |        0.167 | 1544 |
| Credit card (automatic)   |        0.152 | 1522 |
| Electronic check          |        0.453 | 2365 |
| Mailed check              |        0.191 | 1612 |

### TechSupport

| TechSupport         |   churn_rate |    n |
|:--------------------|-------------:|-----:|
| No                  |        0.416 | 3473 |
| No internet service |        0.074 | 1526 |
| Yes                 |        0.152 | 2044 |

### PaperlessBilling

| PaperlessBilling   |   churn_rate |    n |
|:-------------------|-------------:|-----:|
| No                 |        0.163 | 2872 |
| Yes                |        0.336 | 4171 |

## Correlation of numeric features with churn

|                |      y |
|:---------------|-------:|
| tenure         | -0.352 |
| MonthlyCharges |  0.193 |
| TotalCharges   | -0.198 |
| SeniorCitizen  |  0.151 |
