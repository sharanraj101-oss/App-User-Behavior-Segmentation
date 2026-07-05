# 📊 App User Behavior Segmentation & Analytics Dashboard

This is my end-to-end unsupervised machine learning and data science pipeline designed to analyze, segment, and visualize mobile application user behavior. Leveraging **K-Means Clustering** and **Principal Component Analysis (PCA)**, I built this system to segment a dataset of 50,000 users into 4 distinct, actionable personas. The project also features an interactive **Streamlit dashboard** for business targeting, profiling, and exporting filtered user lists.

---

## 🌟 Key Features I Implemented

*   **Complete Data Pipeline**: Seamless workflows for downloading raw datasets, data cleaning (handling duplicates & missing entries), scaling, cluster fitting, and PCA dimensionality reduction.
*   **Optimal Cluster Identification**: Implemented the Elbow Method (within-cluster sum of squares) to find and confirm the optimal number of segments ($k=4$).
*   **Dimensionality Reduction**: Utilized 2D PCA to project the high-dimensional behavior space (10 features) down to 2 principal components for clear visualization.
*   **Interactive Analytics Dashboard**: Built a responsive Streamlit dashboard with three core sections:
    1.  **Dashboard Overview**: High-level KPIs and cluster distribution plots.
    2.  **Segment Visualization (PCA)**: Dynamic, interactive 2D scatter plots of the clusters using Plotly.
    3.  **Targeting & Export**: Interactive filters to target specific segments and export customized user lists directly to CSV.
*   **Persona-Driven Recommendations**: Formulated marketing and product strategies tailored to each of the 4 user segments.

---

## 🧬 User Persona Segments ($k=4$)

Based on my analysis and clustering of behavioral features, I divided the user base into four main personas:

| Cluster | Persona Icon & Name | Core Behavioral Characteristics | Actionable Product/Marketing Strategy |
| :---: | :--- | :--- | :--- |
| **0** | 🔇 **Silent Users** (Non-Raters) | Regular app usage and engagement, but zero ratings submitted (recorded/imputed as `0`). | Trigger targeted in-app rating prompt or survey with a small incentive (e.g., feature unlock). |
| **1** | 🧭 **Active Explorers** (High Page Views) | High volume of pages viewed per session (average **19.3 pages**), actively exploring features. | Contextual promotions, personalized shortcuts, and search recommendations. |
| **2** | ⚡ **Focused Browsers** (Low Page Views) | Minimal pages viewed per session (average **7.7 pages**). Logs in, does a specific task, and leaves. | Keep UI clean, optimized, and path-clear. Offer quick-action widgets and shortcuts. |
| **3** | 🧘 **Immersive Users** (Long Sessions) | Longest continuous session durations (average **29.1 minutes**). Deep interaction with content. | Premium subscription upsells, loyalty programs, long-form content notifications. |

---

## 📁 Project Structure

Here is an overview of the scripts and files I wrote for this project:

*   **[download_data.py](download_data.py)**: Downloads the raw dataset (`user_behavior_dataset.csv`) from Google Drive with automatic virus confirmation token handling.
*   **[preprocess.py](preprocess.py)**: Handles duplicate rows, imputes missing user ratings, selects the 10 core numerical features, and applies `StandardScaler` to prepare the data.
*   **[train_kmeans.py](train_kmeans.py)**: Runs the K-Means clustering algorithm across a range of $k$ (1 to 10), saves the Elbow curve plot, and saves the final trained $k=4$ model to `models/kmeans_model.pkl`.
*   **[run_pca.py](run_pca.py)**: Projects the scaled features onto 2 principal component dimensions for visualization, outputs variance ratios, and saves the 2D cluster scatter plot.
*   **[generate_plots.py](generate_plots.py)**: Script for exploratory data analysis (EDA), generating visualizations of distributions (engagement, churn, ratings, categorical variables) and the correlation matrix.
*   **[profile_clusters.py](profile_clusters.py)**: Quick CLI script to check the mean behavioral and demographic metrics per cluster.
*   **[app.py](app.py)**: The interactive Streamlit dashboard application.
*   **`models/`**: Stores serialized pickled models (`kmeans_model.pkl`, `pca_model.pkl`).
*   **`plots/`**: Stores all generated data visualizations and evaluation graphs.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

It is recommended to use a Python virtual environment (Python 3.8+).

Clone this repository and install the dependencies:

```bash
# Clone the repository (or navigate to the project directory)
cd "App User Behavior"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
```

### 2. Running the Data & ML Pipeline

You can run the scripts in order to download, clean, model, and visualize the data:

```bash
# Step 1: Download the dataset
python download_data.py

# Step 2: Run Exploratory Data Analysis & generate static plots
python generate_plots.py

# Step 3: Clean and scale the dataset
python preprocess.py

# Step 4: Run K-Means modeling & Elbow analysis
python train_kmeans.py

# Step 5: Run PCA for dimensionality reduction and mapping
python run_pca.py

# Step 6: Profile the clusters
python profile_clusters.py
```

### 3. Launching the Interactive Dashboard

Launch the Streamlit dashboard app to interactively explore segments and export targets:

```bash
streamlit run app.py
```

Open the URL provided in your terminal (usually `http://localhost:8501`) to view the dashboard!

---

## 📊 Modeling & Dimensionality Reduction Details

### Behavioral Features Used for Clustering
1.  `sessions_per_week`
2.  `avg_session_duration_min`
3.  `daily_active_minutes`
4.  `feature_clicks_per_session`
5.  `notifications_opened_per_week`
6.  `pages_viewed_per_session`
7.  `days_since_last_login`
8.  `rating_given` (missing values imputed with `0`)
9.  `churn_risk_score`
10. `engagement_score`

### Principal Component Analysis (PCA)
*   **PC1 & PC2 Variance Explained**: Combined components explain a significant portion of the total variance in behavioral metrics, allowing for clear visual separation of the 4 clusters on a 2D grid.
*   Models are persisted using Python's `pickle` library inside the `models/` directory for fast loading within the live Streamlit dashboard.
