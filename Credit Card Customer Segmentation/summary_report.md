# Summary Report — Credit Card Customer Segmentation

**Author:** Herit Tanna | **Project:** Practical Exam – Unsupervised Learning (Set C)

## 1. Business problem and dataset
The credit-cards division of an Indian private bank wants to replace one-size-fits-all card upgrades, rewards and credit-limit revisions with offers tailored to **behavioural segments**. I used the Kaggle *Credit Card Dataset for Clustering*: **8,950 active cardholders** and 18 columns (balance, purchases, cash advance, credit limit, payments, tenure, frequencies). Only `CREDIT_LIMIT` (1 value) and `MINIMUM_PAYMENTS` (313 values) had missing data, filled with the median; there were no duplicates. EDA showed that **22.8 % of customers never purchase**, and that only **~26 % of customers generate 80 % of purchase volume** (a Pareto-type pattern).

## 2. Feature engineering and preprocessing
I added four behavioural features: `Monthly_Avg_Purchase`, `Monthly_Avg_Cash_Advance`, `Limit_Usage` (balance ÷ limit) and `Payment_to_Minpayment_Ratio`. All money columns were extremely right-skewed with large outliers, which would dominate any distance-based algorithm. So I (1) **capped outliers at Q3 + 3×IQR** (no rows dropped), (2) applied **`log1p`**, and (3) used **`StandardScaler`** on all 21 features. PCA showed 9 components keep 90 % variance, but I clustered on the full feature set for interpretability and easy deployment.

## 3. Which algorithm performed best?
| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| **K-Means (k=4)** | 4 | **0.196** | 1.70 | **2,042** |
| Agglomerative (Ward) | 4 | 0.156 | 1.91 | 1,657 |
| DBSCAN (eps 2.5, min_samples 3) | 5 + 1.6 % noise | 0.097 | 0.87* | 8 |

*DBSCAN's low Davies-Bouldin is an artefact of one giant cluster plus tiny 3-customer pockets.
**K-Means ranked first** and was very stable (silhouette 0.1965 ± 0.0001 over five seeds). This matched business intuition: K-Means and Ward both give four balanced, interpretable segments, while DBSCAN, in 21 dimensions with no clear density gaps, behaves mainly as an **outlier detector**. K-Means was deployed because it can also score new customers.

## 4. The four cardholder segments
- **Premium Transactors (19 %)** – highest limit and frequent, high spending; low cash advance; relatively many pay in full.
- **Active Revolvers (26 %)** – high balance (~68 % of limit), almost never pay in full, mix of purchases and cash advance; main interest-income group.
- **Cash-Advance Reliant (26 %)** – almost no purchases but heavy cash advance; card used like a loan, highest credit risk.
- **Low-Balance Light Spenders (29 %)** – small balance, modest spend, low risk, under-used card.

DBSCAN's ~1.6 % noise customers have ~3× the average cash advance, payments and purchases and need manual Relationship-Manager or risk review.

## 5. Recommended actions and next steps
Premium upgrade offers for Premium Transactors; balance-transfer/EMI plans for Active Revolvers; risk monitoring plus cheaper personal loans for Cash-Advance Reliant; cashback and no-cost EMI for Light Spenders.
**Next:** add merchant-category, income/age and credit-bureau data; use attrition/upgrade outcomes for **semi-supervised refinement**; and expose `predict_segment()` as a **real-time scoring API**.
