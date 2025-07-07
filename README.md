# telco-churn-prediction
Churn prediction and Analysis  ML project using Telco data

his project analyzes customer churn data from a telecom company to predict whether a customer is likely to leave the service. The goal is to identify patterns that lead to churn and build a machine learning model to help reduce customer loss.

---

## 📁 Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contains customer information including demographics, contract type, billing, and services used.
- Target variable: **`Churn`** (Yes/No)

---

## 🧹 Data Preprocessing

- Removed customers with `tenure = 0` (newly joined, no behavior pattern yet)
- Converted `TotalCharges` from object to numeric, and handled empty strings
- Encoded categorical variables using `pd.get_dummies()` for one-hot encoding
- Detected **class imbalance**: Churn = 26%, No Churn = 74%
- Applied **SMOTE** oversampling to balance the dataset

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized distribution of churned vs. non-churned customers
- Key findings from EDA:
  - Customers on **Month-to-Month contracts** churn the most
  - **Low tenure** and **high monthly charges** are strong churn indicators
  - Customers without **Tech Support** or **Online Security** are more likely to leave
  - Payment method also influences churn (manual payment = higher churn)

---

## 🤖 Model Used

### ✅ Random Forest Classifier

We chose **Random Forest** because:

- It performs well with both numerical and categorical data
- It's less sensitive to outliers and doesn't need feature scaling
- Provides **feature importance**, helping us understand what drives churn

Other models considered:
- Logistic Regression (too linear, lower performance)
- KNN (not scalable)
- XGBoost (more complex, used for comparison later)

---

## 🧪 Model Performance

- Accuracy: **~79%**
- Improved **Recall for churn class** after balancing data with SMOTE
- Feature importance shows that:
  - `tenure`, `Contract`, `MonthlyCharges`, `OnlineSecurity`, and `TechSupport` are top features

---

## 💡 Business Insights

- Retention strategy should focus on:
  - Month-to-month contract users
  - Customers with low tenure
  - Users without support or security services
- Offer discounts, better onboarding, and bundled services

---

## 📂 How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
