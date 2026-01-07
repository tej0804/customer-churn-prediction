# Customer Churn Prediction (Time-Series)

This project builds a time-series customer churn prediction system using transactional retail data.

## Key Highlights
- Defined churn based on customer inactivity windows
- Engineered behavioral features (RFM, tenure, purchase intensity)
- Addressed class imbalance using class weights and SMOTE
- Compared Logistic Regression, Random Forest, and XGBoost
- Achieved ROC-AUC ~0.80 using XGBoost
- Deployed real-time prediction using Streamlit

## Dataset
The dataset used in this project is **Online Retail II (UCI)**.

Due to size constraints, the raw dataset is not included in the repository.
It can be downloaded from:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci


## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit

## How to Run
```bash
streamlit run app.py
