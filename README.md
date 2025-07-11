# Company Bankruptcy Prediction

## Overview
This project predicts whether a company is likely to go bankrupt using machine learning models. It analyzes financial data from a dataset of companies, employing two feature selection approaches:
- **Correlation-based** (selecting features with high correlation to the target)
- **PCA** (Principal Component Analysis for dimensionality reduction)

The project also addresses class imbalance using **SMOTE** and applies multiple classification models to ensure robust predictions.

---

## Dataset
The dataset (`data_company.csv`) contains financial metrics for **6,819 companies**, with **96 columns**:

- **Target**: `Bankrupt?`  
  - `0`: Not Bankrupt  
  - `1`: Bankrupt  
  - Class distribution: ~96.77% non-bankrupt and ~3.23% bankrupt.

- **Features**: 95 financial indicators (e.g., ROA, Debt Ratio, Net Income to Total Assets, etc.)

✅ No missing values in the dataset.

---

## Features and Methodology

### Exploratory Data Analysis (EDA)
- Visualizes data distributions, outliers, and correlations using:
  - Boxplots
  - Histograms
  - Scatterplots
  - Correlation heatmap
- Confirms class imbalance in the `Bankrupt?` label.

### Preprocessing
- **Correlation-based Approach**:  
  Selects features with absolute correlation > 0.2 to the target.

- **PCA Approach**:  
  Reduces dimensionality to **10 principal components**, capturing significant variance (visualized with a cumulative explained variance plot).

- **SMOTE**:  
  Applies Synthetic Minority Oversampling Technique to balance the dataset.

- **StandardScaler**:  
  Standardizes features for better model performance.

---

## Models Used
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

---

## Handling Class Imbalance
- Models use:
  - `class_weight='balanced'`
  - or `scale_pos_weight=1`
- Applies prediction thresholds (0.5, 0.3, 0.2) to improve **recall** for minority class (`Bankrupt=1`).

---

## Evaluation
- **Metrics**: F1 Score, Accuracy, Recall, Precision, ROC-AUC
- **Visualizations**:
  - Confusion matrices
  - Metric heatmaps
  - Feature importance plots

---

## Predictions
- Tests on **2 random samples** (indices `239` and `2850`, both non-bankrupt)
- Outputs predictions and probabilities in a consolidated DataFrame
- View is pivoted for clarity

📌 **Observations**:
- SVM, Decision Tree, and KNN showed lower performance for bankrupt cases using the Correlation-based approach.
- Performance improved in PCA, especially with lower thresholds (0.3, 0.2).

---

## Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/company-bankrupt-prediction.git](https://github.com/nvquangyn02/Company-Bankrupt-Prediction.git)
cd company-bankrupt-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn
```

Ensure Python **3.6+** is installed.

---

## Usage

1. Place the dataset `data_company.csv` in the project root directory.
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Company_Bankruptcy_Prediction.ipynb
   ```
3. Execute all cells to:
   - Perform EDA
   - Train models
   - Generate predictions

4. Check the output:
   - Visualizations (e.g., correlation heatmap, PCA plot)
   - Evaluation metrics
   - `predictions_2_samples.csv` (output predictions for 2 random samples)

---

## File Structure

```
company-bankrupt-prediction/
├── Company_Bankruptcy_Prediction.ipynb  # Main Jupyter notebook
├── data_company.csv                    # Dataset (not included, must be provided)
├── predictions_2_samples.csv           # Output predictions for 2 samples
├── README.md                           # Project documentation
```

---

## Results

### EDA
- Confirmed class imbalance (3.23% bankrupt)
- Key features: ROA, Debt Ratio, etc.

### Model Performance
- **PCA** approach outperformed Correlation-based in predicting bankrupt cases.
- **Random Forest** and **XGBoost** performed well.
- **SVM**, **KNN**, **Decision Tree** improved with lower thresholds in PCA.

### Predictions
- On test samples (indices 239, 2850; true label: 0), most models predicted correctly.
- For testing on bankrupt cases, modify:
  ```python
  random_indices = [6500, 239]  # 6500 is a known bankrupt sample
  ```

---

## Improving Predictions

To enhance predictions for bankrupt class (1):
- Increase `sampling_strategy` in SMOTE (e.g., 1.5)
- Tune hyperparameters (e.g., `n_estimators=200` for RF, `C=0.5` for SVM)
- Lower prediction threshold further (e.g., 0.1)


