Company Bankruptcy Prediction
Overview
This project predicts whether a company is likely to go bankrupt using machine learning models. It analyzes financial data from a dataset of companies, employing two feature selection approaches: Correlation-based (selecting features with high correlation to the target) and PCA (Principal Component Analysis for dimensionality reduction). The project addresses class imbalance using SMOTE and applies multiple classification models to ensure robust predictions.
Dataset
The dataset (data_company.csv) contains financial metrics for 6,819 companies, with 96 columns:

Target: Bankrupt? (0: Not Bankrupt, 1: Bankrupt), with a class distribution of ~96.77% non-bankrupt and ~3.23% bankrupt.
Features: 95 financial indicators, such as ROA, Debt Ratio, Net Income to Total Assets, etc.
No missing values are present in the dataset.

Features and Methodology

Exploratory Data Analysis (EDA):
Visualizes data distributions, outliers, and correlations using boxplots, histograms, scatterplots, and a correlation heatmap.
Confirms class imbalance in the Bankrupt? label.


Preprocessing:
Correlation-based Approach: Selects features with absolute correlation > 0.2 to the target.
PCA Approach: Reduces dimensionality to 10 principal components, capturing significant variance (visualized with a cumulative explained variance plot).
Applies SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
Standardizes features using StandardScaler.


Models:
Logistic Regression
Random Forest
XGBoost
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree


Handling Class Imbalance:
Uses class_weight='balanced' or scale_pos_weight=1 for models.
Applies prediction thresholds (0.5, 0.3, 0.2) to improve recall for the minority class (Bankrupt=1).


Evaluation:
Metrics: F1 Score, Accuracy, Recall, Precision, ROC-AUC.
Visualizations: Confusion matrices, evaluation metric heatmaps, feature importance plots.


Predictions:
Tests on 2 random samples (indices 239 and 2850, both non-bankrupt) using all models.
Outputs predictions and probabilities in a consolidated DataFrame, with a pivoted view for clarity.
SVM, Decision Tree, and KNN showed lower performance for bankrupt cases in the Correlation-based approach but improved in PCA, especially with lower thresholds (0.3, 0.2).



Installation

Clone the repository:git clone https://github.com/your-username/company-bankrupt-prediction.git
cd company-bankrupt-prediction


Install required Python libraries:pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn


Ensure Python 3.6+ is installed.

Usage

Place the dataset (data_company.csv) in the project root directory.
Run the Jupyter notebook:jupyter notebook Company_Bankruptcy_Prediction.ipynb


Execute all cells to perform EDA, train models, and generate predictions.
Check the output:
Visualizations (e.g., correlation heatmap, PCA variance plot).
Model evaluation metrics and confusion matrices.
Consolidated predictions for 2 random samples in predictions_2_samples.csv.



File Structure
company-bankrupt-prediction/
├── Company_Bankruptcy_Prediction.ipynb  # Main Jupyter notebook
├── data_company.csv                    # Dataset (not included, must be provided)
├── predictions_2_samples.csv           # Output predictions for 2 samples
├── README.md                           # Project documentation

Results

EDA: Confirmed class imbalance (3.23% bankrupt) and identified key features like ROA and Debt Ratio.
Model Performance:
PCA approach generally outperformed Correlation-based for bankrupt predictions.
Random Forest and XGBoost showed strong performance, while SVM, KNN, and Decision Tree struggled with bankrupt cases in the Correlation-based approach.
Lower thresholds (0.3, 0.2) improved recall for bankrupt predictions, especially for SVM and KNN in PCA.


Predictions:
For samples (indices 239, 2850, true label 0), most models correctly predicted non-bankrupt.
To test bankrupt cases, modify random_indices in Section 5 (e.g., random_indices = [6500, 239]) for index 6500 (true label 1).



Improving Predictions
To enhance predictions for bankrupt cases (class 1):

Increase SMOTE sampling_strategy (e.g., 1.5) in the preprocessing section.
Tune model hyperparameters (e.g., n_estimators=200 for Random Forest, C=0.5 for SVM).
Lower the prediction threshold further (e.g., 0.1) in Section 5.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Submit a pull request with a clear description of changes.

Contact
For questions or feedback, open an issue on GitHub or contact [your-email@example.com].
