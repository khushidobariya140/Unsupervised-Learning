# Unsupervised Learning — PR 1
## Mall Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)

> **Red & White Skill Education — Unsupervised Learning Practical Report 1**

---

## 📌 Project Overview

This project applies **unsupervised machine learning** to the **Mall Customer Segmentation Dataset** to discover meaningful customer groups based on demographic and spending behaviour.

Three clustering algorithms are implemented and compared:

- **K-Means Clustering**
- **Agglomerative Hierarchical Clustering**
- **DBSCAN**

The primary clustering analysis uses standardized **Annual Income** and **Spending Score**, while **Age** is also standardized and retained for customer profiling.

The objective is not only to create clusters, but also to understand **why each algorithm produces its result** and translate the discovered segments into practical business insights for mall management.

---

## 🎯 Project Objectives

- Load and understand the Mall Customer Segmentation dataset.
- Perform exploratory data analysis (EDA).
- Clean and prepare the data for clustering.
- Encode the categorical `Gender` feature.
- Apply `StandardScaler` to numerical features.
- Select `Annual_Income` and `Spending_Score` for the primary 2D clustering analysis.
- Determine a suitable number of K-Means clusters using:
  - Elbow Method
  - Silhouette Score
- Apply Agglomerative Hierarchical Clustering using Ward linkage.
- Tune DBSCAN using a k-distance plot and parameter grid search.
- Compare the three clustering algorithms visually and numerically.
- Interpret customer segments from a business perspective.

---

## 📊 Dataset

**Dataset:** Mall Customer Segmentation Data  
**Source:** Kaggle  
**Records:** 200 customers  
**Original columns:** 5

### Dataset Link

[Kaggle — Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

### Original Features

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of USD |
| `Spending Score (1-100)` | Mall spending score |

### Preprocessing

The dataset is transformed as follows:

1. `CustomerID` is removed because it is only an identifier.
2. `Annual Income (k$)` is renamed to `Annual_Income`.
3. `Spending Score (1-100)` is renamed to `Spending_Score`.
4. `Gender` is encoded using `LabelEncoder`.
5. `Age`, `Annual_Income`, and `Spending_Score` are standardized using `StandardScaler`.
6. `df_2f` contains the standardized `Annual_Income` and `Spending_Score` features used for the primary clustering analysis.

---

## 🔎 Exploratory Data Analysis

The notebook performs:

- Dataset preview using `head()`
- Dataset structure inspection using `info()`
- Statistical summary using `describe()`
- Missing-value and duplicate checks
- Histograms with KDE for:
  - Age
  - Annual Income
  - Spending Score
- Pairplot for feature relationships
- Correlation heatmap

### Why Annual Income and Spending Score?

These two features provide a clear 2D representation of customer purchasing behaviour. They allow the clusters to be visualized directly without applying PCA.

---

## ⚖️ Why Feature Scaling?

Clustering algorithms such as **K-Means, DBSCAN, and Hierarchical Clustering** depend on distances between observations.

The numerical features have different ranges. Without scaling, a feature with a larger numerical range could have a stronger influence on distance calculations.

`StandardScaler` transforms the numerical features so that they are on a comparable scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_values = scaler.fit_transform(
    df[['Age', 'Annual_Income', 'Spending_Score']]
)
```

`Gender` is kept as a binary encoded feature and is not scaled.

---

# 🤖 Algorithms

## 1. K-Means Clustering

K-Means groups observations around a specified number of cluster centroids.

### Techniques Used

- Elbow Method
- Silhouette Score
- Final K-Means model
- Cluster centroid visualization
- Cluster profiling

The notebook evaluates `k = 1` to `10` using the Elbow Method and `k = 2` to `10` using the Silhouette Score.

For this dataset, the highest silhouette score occurs at **k = 5**, which also aligns with the typical five-segment solution specified in the practical brief.

### Final Configuration

```python
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

---

## 2. Agglomerative Hierarchical Clustering

Hierarchical clustering builds a hierarchy of observations and represents it using a **dendrogram**.

### Configuration

- Linkage: `ward`
- Number of clusters: `5`

The dendrogram is used to understand how customers are progressively merged into groups.

### Ward Linkage

Ward linkage attempts to minimize the increase in within-cluster variance when two groups are merged.

---

## 3. DBSCAN

DBSCAN is a **density-based clustering algorithm**.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

### Parameter Tuning

The notebook performs:

- 4-nearest-neighbour distance analysis
- k-distance plot
- Grid search over:

```text
eps:
0.2, 0.3, 0.4, 0.5, 0.6

min_samples:
3, 4, 5, 6
```

For each combination, the notebook records:

- Number of clusters
- Number of noise points
- Silhouette Score

### Final Configuration

The selected balanced configuration is:

```text
eps = 0.4
min_samples = 5
```

This produces **4 dense clusters and 15 noise points** for the final DBSCAN analysis.

Noise observations receive the label:

```text
-1
```

---

# 📈 Model Evaluation

The three algorithms are compared using:

### 1. Silhouette Score

Higher values generally indicate better-defined and better-separated clusters.

### 2. Davies-Bouldin Index

Lower values generally indicate better separation between clusters.

### 3. Calinski-Harabasz Index

Higher values generally indicate stronger separation relative to within-cluster dispersion.

> **Important:** DBSCAN evaluation excludes noise points (`-1`) from the metric calculations, as required by the practical specification. Therefore, DBSCAN's metric values should be interpreted with this difference in evaluation population in mind.

---

# 👥 Customer Segments

The K-Means result provides an interpretable five-segment customer structure.

| Segment | Description | Suggested Business Action |
|---|---|---|
| **Average Income, Average Spenders** | Customers with moderate income and spending | Loyalty offers, product bundles and personalized promotions |
| **High Income, High Spenders** | Higher-income customers with high spending | Premium loyalty rewards and exclusive offers |
| **Low Income, High Spenders** | Lower-income customers with relatively high spending | Value bundles, limited-time promotions and loyalty incentives |
| **High Income, Low Spenders** | Higher-income customers with relatively low mall spending | Personalized engagement and premium product discovery |
| **Low Income, Low Spenders** | Lower-income customers with lower spending | Budget promotions and entry-level offers |

These segment descriptions are **descriptive interpretations of the clustering results**. They should not be treated as causal conclusions because the dataset does not contain detailed transaction histories.

---

# 🔬 Algorithm Comparison

| Feature | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Type | Centroid-based | Connectivity-based | Density-based |
| Requires `k` beforehand | Yes | Yes, for final cut | No |
| Detects noise | No | No | Yes |
| Cluster shape | Mainly compact/centroid-based | Depends on linkage | Can detect arbitrary shapes |
| Main parameter | `n_clusters` | `n_clusters`, linkage | `eps`, `min_samples` |
| Visualization | Scatter plot | Dendrogram + scatter | Scatter + noise |
| Main strength | Simple and interpretable | Shows hierarchy | Finds dense regions and noise |

### Key Difference

K-Means assigns every customer to a cluster around a centroid.

Hierarchical clustering builds a tree of relationships between customers.

DBSCAN groups customers according to density and can identify observations that do not belong to a sufficiently dense region.

Because these algorithms use different assumptions, their cluster labels should **not** be compared by label number alone. Cluster profiles and spatial positions are more meaningful.

---

# 💼 Business Insights

The clustering results can help mall management think about customers as different behavioural segments instead of treating all customers identically.

### High Income + High Spenders
Potentially valuable customers for premium loyalty programs, exclusive offers and early-access campaigns.

### Low Income + High Spenders
Customers who spend strongly despite lower income. Value-based promotions and loyalty incentives can be tested.

### High Income + Low Spenders
Customers with purchasing capacity but comparatively lower spending scores. Personalized engagement and product discovery may help encourage additional spending.

### Low Income + Low Spenders
Budget-oriented campaigns, discounts and entry-level offers can be considered.

### Average Income + Average Spenders
This mainstream segment can be targeted using general loyalty programs, bundles and personalized promotions.

---

# 🖼️ Visualizations

The notebook includes:

- Age distribution
- Annual Income distribution
- Spending Score distribution
- Pairplot
- Correlation heatmap
- K-Means Elbow Method
- K-Means Silhouette Score
- K-Means cluster visualization
- K-Means centroid visualization
- Hierarchical dendrogram
- Hierarchical cluster visualization
- K-Means vs Hierarchical comparison
- DBSCAN k-distance plot
- DBSCAN parameter grid-search results
- DBSCAN cluster visualization
- Three-algorithm comparison

The visualizations use a **clean, minimal and soft pastel style** for consistent presentation.

---

# 📁 Project Structure

```text
UL-PR1-Mall-Customer-Segmentation/
│
├── UL_PR1.ipynb
├── UL_PR1.html
├── Mall_Customers.csv
├── Mall_Customers_Clustered.csv
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── eda_pairplot.png
    ├── correlation_heatmap.png
    ├── elbow_method.png
    ├── silhouette_score.png
    ├── kmeans_clusters.png
    ├── hierarchical_dendrogram.png
    ├── hierarchical_clusters.png
    ├── dbscan_k_distance.png
    ├── dbscan_clusters.png
    └── algorithm_comparison.png
```

---

# 🛠️ Technologies & Libraries

### Programming Language

- Python

### Development Environment

- Jupyter Notebook
- VS Code / Jupyter

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

# ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project folder

```bash
cd UL-PR1-Mall-Customer-Segmentation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
UL_PR1.ipynb
```

### 6. Run all cells

Use:

```text
Kernel → Restart & Run All
```

The notebook should execute from beginning to end without errors.

---

# 📄 Files Description

| File | Purpose |
|---|---|
| `UL_PR1.ipynb` | Complete executable Jupyter Notebook |
| `UL_PR1.html` | HTML version of the notebook |
| `Mall_Customers.csv` | Original dataset |
| `Mall_Customers_Clustered.csv` | Dataset containing final clustering labels |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `screenshots/` | Important project visualizations |

---

# 🎥 Project Video

**Video:** `[Add your Google Drive / YouTube unlisted link here]`

The video should demonstrate:

- Feature scaling rationale
- EDA
- Elbow Method
- Silhouette Score
- K-Means clustering
- Dendrogram and Ward linkage
- DBSCAN parameters
- k-distance plot
- Algorithm comparison
- Business insights

---

# 🎓 Practical Report

**Institute:** Red & White Skill Education  
**Subject:** Unsupervised Learning  
**Practical:** PR 1  
**Algorithms:** K-Means, Agglomerative Hierarchical Clustering, DBSCAN  
**Dataset:** Mall Customer Segmentation Dataset  
**Dataset Size:** 200 customer records

---

# ✅ Submission Checklist

- [x] Dataset loaded and inspected
- [x] CustomerID removed
- [x] Columns renamed
- [x] Gender encoded
- [x] StandardScaler applied
- [x] `df_scaled` created
- [x] `df_2f` created
- [x] EDA completed
- [x] Pairplot and correlation heatmap
- [x] K-Means Elbow Method
- [x] K-Means Silhouette Score
- [x] Final K-Means clustering
- [x] Cluster profiles and business interpretation
- [x] Hierarchical dendrogram
- [x] Agglomerative clustering
- [x] K-Means vs Hierarchical comparison
- [x] DBSCAN k-distance analysis
- [x] DBSCAN parameter grid search
- [x] Final DBSCAN clustering
- [x] Noise-point visualization
- [x] Three-algorithm comparison
- [x] Clustering evaluation metrics
- [x] Business insights
- [x] HTML notebook
- [x] Requirements file
- [ ] Add final screenshots to `/screenshots`
- [ ] Add video link
- [ ] Create 3+ meaningful Git commits
- [ ] Perform final Restart & Run All before submission

---

## 👩‍💻 Author

**Khushi Dobariya**

BCA — Artificial Intelligence / Machine Learning & Data Science

---

## ⭐ Project Summary

This project demonstrates how unsupervised learning can be used to discover customer segments without predefined target labels. K-Means, Hierarchical Clustering and DBSCAN are applied to the same customer dataset, allowing their assumptions, clustering behaviour, strengths and limitations to be compared.

The final analysis converts the discovered clusters into understandable customer segments and practical marketing actions for a mall management scenario.
