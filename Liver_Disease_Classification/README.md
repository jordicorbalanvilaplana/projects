# Liver Disease Classification

A machine learning project for binary classification of liver disease patients using traditional supervised learning models. This project was developed as part of a **Kaggle competition** for the Machine Learning course in the Bachelor's degree in Data Science and Engineering. **Relevant information is summarized below and in the report file above.** 

## Authors

- **Ona Siscart Noguer** — Universitat Politècnica de Catalunya (UPC), Barcelona
- **Jordi Corbalan Vilaplana** — Universitat Politècnica de Catalunya (UPC), Barcelona

---

## Abstract

Early detection of liver disease is essential for improving clinical outcomes and reducing pressure on healthcare systems. This project addresses the problem as a binary classification task using clinical and demographic data. Several supervised learning models were evaluated under experimental conditions, with a focus on understanding their behavior under data constraints such as class imbalance, limited sample size, and potential feature redundancy.

The methodology included exploratory data analysis, feature engineering, class imbalance mitigation, and the use of cross-validation for robust evaluation. Preprocessing strategies were designed to adapt to model-specific requirements, while performance assessment prioritized the F1-score to balance precision and recall—particularly important given the clinical need to correctly identify sick patients.

Despite time and model restrictions, results suggest that even simple models can perform competitively when properly tuned and evaluated. The project also highlights the trade-offs between model complexity, interpretability, and performance in sensitive domains like healthcare.

---

## Problem Statement

Given a database of biometric variables from different patients, the goal is to determine whether each individual should be classified as **healthy (1)** or **sick (0)**. This binary classification task aims to support medical decision-making through automated and accurate predictions.

---

## Dataset

The dataset was obtained from the UCI Machine Learning Repository, containing clinical and demographic data collected from patients in the North East region of Andhra Pradesh, India.

| Property | Description |
|----------|-------------|
| **Samples** | 579 observations (after preprocessing) |
| **Classes** | 414 liver patients, 165 healthy subjects |
| **Features** | 10 numerical clinical variables |
| **Train/Test Split** | 80% training (463 samples), 20% test (116 samples) |

### Clinical Features

| Feature | Description |
|---------|-------------|
| Age | Age of the patient |
| Female | Gender (1 = Female, 0 = Male) |
| TB | Total Bilirubin |
| DB | Direct Bilirubin |
| Alkphos | Alkaline Phosphatase |
| Sgpt | Alanine Aminotransferase (ALT) |
| Sgot | Aspartate Aminotransferase (AST) |
| TP | Total Proteins |
| ALB | Albumin |
| A/G | Albumin to Globulin Ratio |

---

## Methodology

### Exploratory Data Analysis

Initial exploration revealed two main challenges:
1. **Class imbalance**: Significantly more liver patients than healthy subjects
2. **Skewed distributions**: Several variables show non-normal distributions with varying value ranges

Correlation analysis identified redundancy between certain features (e.g., DB/TB correlation of 0.85, Sgpt/Sgot correlation of 0.91), informing subsequent feature engineering decisions.

### Feature Engineering

New features were designed to compress correlated variables and enrich the dataset:

- **B-ratio (DB/TB)**: Proportion of direct bilirubin relative to total, summarizing liver function efficiency
- **SGPT/SGOT ratio**: Ratio between liver enzymes ALT and AST to reduce redundancy
- **Globulins (TP - ALB)**: Non-albumin protein content, indicative of inflammation or immune response
- **Log-Likelihood features**: Class density approximations using Kernel Density Estimation (KDE)

### Data Preprocessing

Different preprocessing pipelines were implemented depending on model requirements:

- **Transformations**: Box-Cox scaling for models assuming normality; standard scaling for others
- **Outlier handling**: IQR-based analysis identified many outliers, attributed to class imbalance; no removal applied due to limited dataset size
- **Class imbalance**: Addressed via `class_weight='balanced'` parameter rather than SMOTE, as synthetic oversampling did not improve performance

### Models Evaluated

| Model Type | Models |
|------------|--------|
| Generative | LDA, QDA, Gaussian Naive Bayes |
| Discriminative | Logistic Regression, SVM (linear, RBF, polynomial kernels) |
| Non-parametric | K-Nearest Neighbors |
| Tree-based | Decision Trees, Random Forest, Extra Trees |
| Ensembles | Hard/Soft Voting, Stacking |

### Evaluation Strategy

- **Primary metric**: Macro-averaged F1-score (appropriate for imbalanced datasets)
- **Validation**: Stratified 10-fold cross-validation to preserve class distribution
- **Secondary considerations**: Per-class F1 scores, precision (to minimize false "healthy" predictions)

---

## Results

| Model | F1 Macro (CV) | F1 Macro (Train) |
|-------|---------------|------------------|
| Hard Voting Ensemble | 0.652 | 0.699 |
| Gaussian Naive Bayes | 0.645 | 0.654 |
| Random Forest | 0.644 | 0.749 |
| Extra Trees | 0.644 | 0.787 |
| Logistic Regression | 0.634 | 0.640 |
| SVC (polynomial) | 0.634 | 0.702 |
| QDA | 0.627 | 0.688 |

**Selected Model**: Gaussian Naive Bayes was chosen as the final model despite the voting ensemble achieving a marginally higher score (+0.007). The rationale was that GNB offers significantly greater simplicity and interpretability with minimal performance trade-off—an important consideration for clinical applications where model transparency is valuable.

---

## Key Findings

1. **Simple models can be competitive**: With proper preprocessing and tuning, Gaussian Naive Bayes achieved nearly the best performance
2. **Class imbalance handling**: Model-internal weighting (`class_weight='balanced'`) outperformed SMOTE oversampling
3. **Feature engineering impact**: Ratio-based features improved performance, but removing original correlated features degraded results
4. **Ensemble limitations**: Stacking performed poorly; voting provided only marginal gains over the best single model

---

## Project Structure

```
├── best.ipynb                              # Main notebook with full ML pipeline
├── Liver patient classification report.pdf # Detailed project report (IEEE format)
├── data/                                   # Dataset files
│   ├── train_features_ILDS.csv
│   ├── train_labels_ILDS.csv
│   ├── test_data_ILDS.csv
│   └── ...
├── models/                                 # Trained model files (.pkl)
├── gridsearch/                             # Hyperparameter search results
└── submissions/                            # Kaggle competition submissions
```

---

## Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
scipy
joblib
```

---

## Usage

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `best.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the analysis

For a comprehensive discussion of methodology, experiments, and results, please refer to the **project report** (`Liver patient classification report.pdf`).

---

## References

1. Kankal, S.S., & Kasar, S. (2025). *A Systematic Review on Machine Learning Techniques for Liver Disease Detection*. Biomedical Materials & Devices.
2. Livers (2021). *Liver Disease Detection Using Machine Learning Techniques*. Livers, 1(4), 294–312.
3. Giannini, E.G., Testa, R., & Savarino, V. (2005). *Liver enzyme alteration: a guide for clinicians*. CMAJ, 172(3), 367–379.

---

## License

This project was developed for educational purposes as part of the Machine Learning course at Universitat Politècnica de Catalunya (UPC).
