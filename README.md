# Predictive Behavior Modeling (Kosinski et al. Replication)

An end-to-end data mining and machine learning pipeline replicating and extending the analytical methodology from *"Mining Big Data to Extract Patterns and Predict Real-Life Outcomes"* (Kosinski, Wang, Lakkaraju, & Leskovec).

## Project Overview
This project investigates how high-dimensional behavioral footprint data can be compressed and modeled to predict real-world user attributes. The pipeline evaluates linear and non-linear supervised learning algorithms across full and reduced feature spaces while preventing data leakage through strict train/validation/test partitioning.

## Methodology & Pipeline
* Exploratory Data Analysis (EDA):** Checked for missing values, analyzed feature sparsity and distributions, inspected inter-feature correlations, and verified observation independence.
* Dimensionality Reduction:** Applied **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)** to compress high-dimensional sparse matrices into dense latent representations while preserving variance.
* Data Partitioning & Model Selection:** Implemented a **3-way data split** (Training, Validation, and Test sets) alongside cross-validation for hyperparameter tuning, followed by retraining optimal configurations on combined training and validation data before final holdout evaluation.
* Classification Models:** Implemented, tuned, and compared:
  * Logistic Regression**
  * Support Vector Machines (SVM) (linear and kernel formulations)
  * k-Nearest Neighbors (k-NN)**

## Key Results & Findings
* Best Generalization:** **Logistic Regression** achieved the strongest generalization performance with **73.08% training accuracy** and a **73.04% cross-validation score**, demonstrating high stability on reduced latent features.
* Overfitting Analysis:** **k-NN** reached **100% training accuracy** but dropped to **69.69% cross-validation accuracy**, highlighting sensitivity to local noise in high-dimensional behavioral data compared to regularized linear boundaries.
* Statistical Validation:** Evaluated classifier performance differences using statistical significance testing to confirm that model improvements reflected genuine underlying patterns rather than sampling variance.

## Repository Structure
* `02_DataMiningCapstone.ipynb` — Complete Python implementation covering data preprocessing, SVD/PCA dimensionality reduction, model training, hyperparameter tuning, and evaluation.

## Tech Used
* Language:** Python 3
* Libraries:** `scikit-learn`, `NumPy`, `pandas`, `SciPy`, `Matplotlib`, `Seaborn`

## Reference
* Kosinski, M., Wang, Y., Lakkaraju, H., & Leskovec, J. (2016). *Mining big data to extract patterns and predict real-life outcomes.* Psychological Methods, 21(4), 493–506.
