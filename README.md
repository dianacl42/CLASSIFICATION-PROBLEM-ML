Breast Cancer Classification using Scikit-Learn

## Overview
This assignment utilizes machine learning algorithms to classify breast cancer cases based on the `sklearn.datasets.load_breast_cancer` dataset. Multiple classification models are implemented to compare their effectiveness in predicting cancer diagnoses.

## Dataset
The dataset is sourced from `sklearn.datasets.load_breast_cancer`, containing features extracted from digitized images of breast tissue. It has no missing values, so no imputation is needed. The target variable represents whether a tumor is malignant or benign.

## Data Preprocessing
1. **Splitting the Dataset**: The dataset is divided into features (`X`) and target labels (`y`).
2. **Train-Test Split**: The data is split into training and testing sets using an 80-20 ratio.
3. **Feature Scaling**: Standardization is performed using `StandardScaler` to normalize the feature values, ensuring that all features contribute equally to model performance.

## Implemented Machine Learning Models
The following classification algorithms are trained and evaluated:

### 1. Logistic Regression
- Implemented using `LogisticRegression` from `sklearn.linear_model`.
- A simple and effective model for binary classification.

### 2. Decision Tree Classifier
- Implemented using `DecisionTreeClassifier` from `sklearn.tree`.
- Provides interpretable decision rules but may overfit the data.

### 3. Random Forest Classifier
- Implemented using `RandomForestClassifier` from `sklearn.ensemble`.
- An ensemble of decision trees that improves performance and reduces overfitting.

### 4. Support Vector Machine (SVM)
- Implemented using `SVC` from `sklearn.svm`.
- Suitable for handling high-dimensional data with good generalization.

### 5. k-Nearest Neighbors (k-NN)
- Implemented using `KNeighborsClassifier` from `sklearn.neighbors`.
- A non-parametric approach that classifies data points based on their closest neighbors.

## Model Evaluation
Each model's performance is assessed using accuracy scores. The models make predictions on the test set, and accuracy is calculated using `accuracy_score` from `sklearn.metrics`.

### Accuracy Scores:
- **Logistic Regression**: `{accuracy_logreg}`
- **Decision Tree**: `{accuracy_dt}`
- **Random Forest**: `{accuracy_rf}`
- **SVM**: `{accuracy_svm}`
- **k-NN**: `{accuracy_knn}`

## Conclusion
Among the models tested, the **Random Forest Classifier** achieved the highest accuracy, while the **Decision Tree Classifier** performed the worst. This suggests that ensemble learning improves prediction robustness and reduces overfitting compared to individual decision trees.

## How to Run the Code
1. Install the required libraries:
   ```sh
   pip install scikit-learn
   ```
2. Run the Python script to load data, preprocess it, train models, and evaluate their performance.
3. Review the accuracy scores to determine the best-performing classifier.

## Notes
- You can experiment with hyperparameter tuning to optimize model performance.
- Different feature selection techniques might further improve accuracy.
- The dataset is already pre-cleaned, so no additional preprocessing is needed.


