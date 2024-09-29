
# MasterSoft - Customer Churn Prediction

## End-to-End Machine Learning Project

## Overview
This project aims to predict customer churn using machine learning models. The dataset contains information about customer demographics, service plans, billing, and churn status, which is analyzed and used to train various classification models. The best-performing model is deployed as an API for real-time predictions.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering and Scaling](#feature-engineering-and-scaling)
- [Model Training](#model-training)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Feature Importance](#feature-importance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Business Insights and Recommendations](#business-insights-and-recommendations)
- [Model Deployment](#model-deployment)

## Dataset Description
- **Rows:** 360
- **Columns:** 21
- **Target Column:** `Churn` (Yes/No)
  
This dataset contains customer data, including demographic information, service details, and billing information. The data is used to understand the reasons for customer churn and create machine learning models for churn prediction.

### Column Information
The dataset consists of the following columns:
- `customerID` (object): Unique identifier for the customer.
- `gender`, `Partner`, `Dependents`, `PhoneService`, etc. (object): Categorical columns representing various customer features.
- `SeniorCitizen`, `tenure` (int64): Numeric columns representing customer attributes.
- `MonthlyCharges`, `TotalCharges` (float64): Numeric columns related to billing information.
- `Churn` (object): Target column with one missing value that was removed.

## Data Preprocessing
1. **Handling Missing Values:**
   - Removed one missing value in the `Churn` column (at index 359).

2. **Handling Non-Numeric Data in `TotalCharges`:**
   - Converted `TotalCharges` to `float64` and checked for non-numeric entries.
   - If necessary, missing values would be imputed using mean, median, or mode, but in this case, no non-numeric values were found.

## Exploratory Data Analysis
### Tenure vs. Churn
- Customers with tenures below 20 months are highly prone to churn. This indicates the importance of early retention strategies.
- After 60 months, the churn rate drops significantly, suggesting that long-tenured customers are more loyal.

### Gender vs. Churn, TotalCharges vs. Churn, and More
- Univariate and bivariate analyses were conducted to explore relationships between customer attributes and churn behavior.

## Feature Engineering and Scaling
### Feature Encoding
- **Categorical Features:** Converted categorical variables (e.g., `gender`, `Partner`) into numerical format using One-Hot Encoding or Label Encoding.

### Feature Scaling
- Applied **StandardScaler** or **MinMaxScaler** to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) for normalization.

### New Feature Creation
- Created a feature combining `StreamingTV` and `StreamingMovies`.
- Created an `average_monthly_charge` feature (`TotalCharges` / `tenure`).

## Model Training
### Train-Test Split
- Split the dataset into training and testing sets using an 80-20 split to ensure reproducibility.

### Model Selection
The following models were trained to predict customer churn:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**

## Model Performance Evaluation
| Model                 | Accuracy | Precision | Recall | F1-score | ROC-AUC Score |
|-----------------------|----------|-----------|--------|----------|---------------|
| Logistic Regression   | 77.78%   | 61.54%    | 42.11% | 50.00%   | 75.47%        |
| Random Forest         | 76.39%   | 60.00%    | 31.58% | 41.38%   | 73.98%        |
| Gradient Boosting     | 76.39%   | 60.00%    | 31.58% | 41.38%   | 66.33%        |
| Support Vector Machine (SVM) | **80.56%** | **66.67%** | **52.63%** | **58.82%** | **76.17%** |
| K-Nearest Neighbors   | 77.78%   | 61.54%    | 42.11% | 50.00%   | 74.68%        |

**Best Model:** The SVM model performed the best, achieving the highest accuracy (80.56%), precision (66.67%), and recall (52.63%).

## Feature Importance
The most important features contributing to churn prediction (using Random Forest and Gradient Boosting) are:
1. `Contract`
2. `MonthlyCharges`
3. `TechSupport`
4. `average_monthly_charge`
5. `TotalCharges`
6. `tenure`
7. `PaymentMethod`
8. `OnlineSecurity`
9. `SeniorCitizen`
10. `PaperlessBilling`

## Hyperparameter Tuning
**Method:** GridSearchCV was used for hyperparameter tuning to improve model performance.

**Tuned Hyperparameters:**
- `C = 10`: Low regularization for fitting complex patterns.
- `gamma = 1`: Medium influence for decision boundaries.
- `kernel = 'linear'`: Data separation using a linear hyperplane.
- `probability = True`: Enabled probability estimates for better decision-making.

## Business Insights and Recommendations
### Business Usefulness
- The SVM model can help predict potential churners, allowing the business to take proactive actions like offering incentives or personalized support.
- Features like `Contract`, `MonthlyCharges`, `TechSupport`, and `TotalCharges` highlight areas needing improvement to reduce churn.

### Recommendations
1. **Contract Type:** Offer discounts to secure longer contracts, as customers with longer contracts are less likely to churn.
2. **Monthly Charges:** Consider discounted bundles or individualized pricing for at-risk customers.
3. **TechSupport and OnlineSecurity:** Provide free trials or discounts to enhance customer satisfaction.
4. **Loyalty Programs:** Reward loyal customers to improve retention.
5. **Payment Options:** Market PaperlessBilling and offer flexible payment options to customers with higher `TotalCharges`.

## Model Deployment
### Saving the Trained Model
- The best-trained SVM model was saved as a `.pkl` file for future use.

### Flask API for Model Deployment
A simple Flask application was built to accept input data through an API and return churn predictions.

- **Route:** `/predict`
- **Deployment Platform:** The Flask app was deployed on Render.com.

**Deployed Link:** [https://mastersoft.onrender.com/](https://mastersoft.onrender.com/)

## Usage
1. Clone the repository.
2. Install the required packages from `requirements.txt`.
3. Run the Flask app locally using `python app.py`.
4. Use the deployed API for predictions by sending POST requests to `/predict` with the necessary input data.

---

