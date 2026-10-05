# 💳 Credit Card Customer Segmentation — Unsupervised Learning

> Grouping ~8,950 credit-card holders of an Indian private bank into **behavioural segments** with **K-Means, Agglomerative Hierarchical Clustering and DBSCAN**, and turning the clusters into **actionable business personas**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-orange) ![Status](https://img.shields.io/badge/Notebook-fully%20executed-brightgreen)

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Video Walkthrough](#-video-walkthrough)
4. [Repository Structure](#-repository-structure)
5. [Setup & How to Run](#-setup--how-to-run)
6. [Methodology](#-methodology)
7. [Results](#-results)
8. [Cluster Personas](#-cluster-personas)
9. [Screenshots](#-screenshots)
10. [Using the Saved Model](#-using-the-saved-model)
11. [Key Insights & Future Work](#-key-insights--future-work)

---

## 🎯 Project Overview
**Business problem:** The credit-cards division of a large private bank (think HDFC / ICICI) wants to stop offering the same card upgrade, reward programme or limit revision to everybody.
They want to know **which type of customer should get which offer**, so that Relationship Managers can reduce attrition and increase wallet share.

**My task (as a Junior Data Scientist):** engineer behavioural features from raw card-usage data, cluster the customers with three unsupervised algorithms, compare them with internal metrics, and explain each segment in business language.

## 📂 Dataset
- **Name:** Credit Card Dataset for Clustering
- **Link:** https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
- **Size:** 8,950 cardholders × 18 columns (balance, purchases, cash advance, credit limit, payments, tenure, frequencies, etc.)
- **Cleaning:** `CUST_ID` dropped; 1 missing `CREDIT_LIMIT` and 313 missing `MINIMUM_PAYMENTS` filled with the median; no duplicate rows.
- The CSV (`CC_GENERAL.csv`) is included in this repo so the notebook runs without downloading anything.

## 🎥 Video Walkthrough
**▶️ Watch the video explanation here:** [PASTE YOUR GOOGLE DRIVE / YOUTUBE (UNLISTED) LINK HERE](PASTE_LINK_HERE)

## 🗂 Repository Structure
```
creditcard-segmentation-unsupervised-learning/
├── CreditCardSegmentation_UnsupervisedLearning.ipynb   # fully executed notebook (all steps, plots, outputs)
├── CC_GENERAL.csv                                      # dataset
├── cc_scaler.pkl                                       # saved StandardScaler
├── cc_segmentation_model.pkl                           # best model (K-Means, k=4)
├── cc_preprocessing.pkl                                # cap limits, medians, feature order, persona names
├── predict_segment.py                                  # stand-alone scoring function for a new cardholder
├── summary_report.md                                   # ~450-word business summary
├── requirements.txt                                    # all libraries
├── README.md                                           # this file
└── images/                                             # all plots (+ interactive 3D HTML)
```

## ⚙️ Setup & How to Run
```bash
# 1. clone the repository
git clone https://github.com/<your-username>/creditcard-segmentation-unsupervised-learning.git
cd creditcard-segmentation-unsupervised-learning

# 2. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows   |   source venv/bin/activate  (Mac/Linux)

# 3. install the libraries
pip install -r requirements.txt

# 4. open the notebook and run all cells (Kernel -> Restart & Run All)
jupyter notebook CreditCardSegmentation_UnsupervisedLearning.ipynb
```
Everything uses fixed random seeds (`random_state=42`), so the results are reproducible. The notebook already contains all outputs, so you can also just read it on GitHub.

## 🔬 Methodology
| Step | What was done |
|---|---|
| **1. EDA** | Shape/info, missing-value imputation, histograms (log axis), correlation heatmap, purchase-utilisation, top cash-advance customers, **Pareto (80/20) check** |
| **2. Preprocessing** | 4 engineered features (`Monthly_Avg_Purchase`, `Monthly_Avg_Cash_Advance`, `Limit_Usage`, `Payment_to_Minpayment_Ratio`) → IQR **capping** (Q3 + 3×IQR, no rows dropped) → `log1p` on money columns → `StandardScaler` (21 features) |
| **3. K-Means** | Elbow + Silhouette for k = 2…10 → **k = 4** → 2D/3D plots → cluster profile → personas |
| **4. Hierarchical** | Ward dendrogram with cut line → compared Ward / Complete / Average linkage → **Ward** chosen → profile & comparison with K-Means |
| **5. DBSCAN** | k-NN distance plot (ε ≈ 2.4) → grid search over ε and min_samples → heatmap → noise analysis |
| **6. Comparison** | Silhouette, Davies-Bouldin, Calinski-Harabasz table; K-Means stability over 5 seeds; business recommendation |
| **7. Deployment** | Saved scaler + model with `joblib`, `predict_segment()` tested on 5 new customers |

## 📊 Results
| Algorithm | Final hyperparameters | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Noise % |
|---|---|---|---|---|---|---|
| **K-Means** ✅ | k = 4, k-means++, n_init = 20 | 4 | **0.196** | 1.70 | **2,042** | 0 % |
| Agglomerative | Ward, 4 clusters | 4 | 0.156 | 1.91 | 1,657 | 0 % |
| DBSCAN | eps = 2.5, min_samples = 3 | 5 | 0.097 | 0.87* | 8 | 1.6 % |

\*DBSCAN's low Davies-Bouldin is an artefact: it found one giant cluster (8,791 customers) and four tiny 3-customer pockets, and its metrics ignore noise points.

**Final ranking: 1) K-Means → 2) Agglomerative (Ward) → 3) DBSCAN.**
K-Means is also **very stable**: silhouette over 5 different seeds = 0.1965 ± 0.0001.
**Pareto check:** only ≈ **26 %** of customers generate **80 %** of all purchase volume.

## 🧑‍💼 Cluster Personas
The deployed K-Means (k = 4) model finds four segments (Ward hierarchical clustering recovers the same four):

| Persona | Size | Profile | Recommended action |
|---|---|---|---|
| **💎 Premium Transactors** | 19 % | Highest limit (~₹6,800) and highest purchases (~3,200), spend almost every month, ≈ 31 % pay in full, low cash advance | Pre-approved premium card upgrade + limit increase, reward multipliers |
| **🔁 Active Revolvers** | 26 % | Highest balance (~2,700, ≈ 68 % of limit used), almost never pay in full (2 %), mixed purchases + cash advance — main interest-income group | Balance-transfer / EMI-conversion plan at lower rate |
| **💸 Cash-Advance Reliant** | 26 % | Almost zero purchases, heavy cash advance (~2,060), ≈ 59 % limit used — card used like a loan, highest credit risk | Risk watch-list, cheaper personal-loan alternative, repayment reminders |
| **🌱 Low-Balance Light Spenders** | 29 % | Tiny balance (~145), modest purchases (~450), ≈ 28 % pay in full — safe but under-used | Spend-based cashback & no-cost EMI offers to grow wallet share |

**DBSCAN as an outlier screen:** the ~1.6 % "noise" customers have ~3× the average cash advance, payments and purchases — they should be reviewed manually by Relationship Managers / the risk team rather than handled by automated campaigns.

## 🖼 Screenshots
| | |
|---|---|
| ![Correlation heatmap](images/03_correlation_heatmap.png) | ![Pareto](images/05_pareto_curve.png) |
| **Correlation heatmap** | **Pareto curve (80/20 check)** |
| ![Elbow and silhouette](images/11_kmeans_elbow_silhouette.png) | ![K-Means 3D](images/14b_kmeans_3d_personas.png) |
| **K-Means: Elbow + Silhouette** | **K-Means 3D clusters with personas** |
| ![Dendrogram](images/15_dendrogram.png) | ![kNN distance](images/18_knn_distance_plot.png) |
| **Dendrogram with cut line** | **k-NN distance plot for DBSCAN ε** |
| ![DBSCAN grid](images/19_dbscan_grid_heatmap.png) | ![Metrics](images/22_metrics_comparison.png) |
| **DBSCAN grid-search heatmap** | **Algorithm comparison** |

> An interactive, rotatable 3D version of the K-Means plot is in `images/kmeans_3d_interactive.html` (download and open in a browser).

## 🔮 Using the Saved Model
```python
from predict_segment import predict_segment

new_customer = dict(BALANCE=1500, BALANCE_FREQUENCY=1.0, PURCHASES=6000, ONEOFF_PURCHASES=3500,
                    INSTALLMENTS_PURCHASES=2500, CASH_ADVANCE=0, PURCHASES_FREQUENCY=1.0,
                    ONEOFF_PURCHASES_FREQUENCY=0.8, PURCHASES_INSTALLMENTS_FREQUENCY=0.7,
                    CASH_ADVANCE_FREQUENCY=0.0, CASH_ADVANCE_TRX=0, PURCHASES_TRX=60,
                    CREDIT_LIMIT=9000, PAYMENTS=6200, MINIMUM_PAYMENTS=500, PRC_FULL_PAYMENT=0.4, TENURE=12)

print(predict_segment(new_customer))     # -> (3, 'Premium Transactors')
```
The function applies the **same pipeline as training** (engineered features → capping → `log1p` → scaling → K-Means).
> The `.pkl` files were created with scikit-learn 1.8; use a similar version (`pip install -r requirements.txt`) to load them without warnings.

## 💡 Key Insights & Future Work
- **Pareto holds:** ~26 % of customers drive 80 % of purchase volume.
- **Extreme cash-advance users are not one type** — some never purchase, others are also heavy spenders — so single-column rules are not enough; multi-feature clustering is needed.
- **Scaling and transformation matter:** all three algorithms are distance-based, so capping, `log1p` and `StandardScaler` were essential.
- **DBSCAN struggles** in 21 dimensions with continuous behaviour (no clear density gaps) but is a useful **outlier detector**.
- **Next steps:** add merchant-category, income/age and credit-bureau data; semi-supervised refinement using attrition/upgrade outcomes; expose `predict_segment()` as a real-time scoring API (e.g. FastAPI).

---
👤 **Author:** Herit Tanna · Diploma in Computer Engineering (GTU) · Entry-level Data Analyst
📝 *Practical Exam – Unsupervised Learning (Set C)*
