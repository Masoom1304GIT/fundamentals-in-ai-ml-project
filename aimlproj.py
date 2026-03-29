# ==========================================
# AI Loan Approval System
# Creator: Masoom Ali
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data():
    data = {
        'Income': [25000,40000,50000,60000,20000,80000,30000,70000,100000,120000,
                   45000,55000,65000,75000,85000,95000,110000,130000],

        'CreditScore': [600,650,700,720,580,750,640,710,780,800,
                        660,690,710,730,760,770,790,810],

        'Age': [25,35,45,50,23,40,30,48,55,60,
                28,38,42,47,52,57,62,65],

        'LoanAmount': [100000,200000,150000,250000,120000,300000,180000,270000,350000,400000,
                       160000,210000,240000,260000,290000,320000,370000,420000],

        'Employment': [0,1,1,1,0,1,0,1,1,1,
                       1,1,1,1,1,1,1,1],

        'Approved': [0,1,1,1,0,1,0,1,1,1,
                     1,1,1,1,1,1,1,1]
    }

    return pd.DataFrame(data)


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def preprocess_data(df):
    df['Income_to_Loan'] = df['Income'] / df['LoanAmount']
    return df


# ==========================================
# 3. TRAIN MODEL
# ==========================================
def train_model(df):
    X = df[['Income', 'CreditScore', 'Age', 'LoanAmount', 'Employment', 'Income_to_Loan']]
    y = df['Approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X


# ==========================================
# 4. EVALUATE MODEL
# ==========================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n===== MODEL PERFORMANCE =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ==========================================
# 5. VISUALIZATION
# ==========================================
def visualize_data(df, model, X):

    # Income vs Approval
    plt.figure()
    plt.hist(df[df['Approved'] == 1]['Income'], alpha=0.7, label="Approved")
    plt.hist(df[df['Approved'] == 0]['Income'], alpha=0.7, label="Rejected")
    plt.title("Income vs Loan Approval")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    # Credit Score vs Approval
    plt.figure()
    plt.hist(df[df['Approved'] == 1]['CreditScore'], alpha=0.7, label="Approved")
    plt.hist(df[df['Approved'] == 0]['CreditScore'], alpha=0.7, label="Rejected")
    plt.title("Credit Score vs Loan Approval")
    plt.xlabel("Credit Score")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    # Feature Importance
    plt.figure()
    plt.bar(X.columns, model.feature_importances_)
    plt.title("Feature Importance")
    plt.xticks(rotation=30)
    plt.show()


# ==========================================
# 6. USER PREDICTION
# ==========================================
def predict_user(model):
    print("\n===== ENTER USER DETAILS =====")

    income = float(input("Income: "))
    credit = float(input("Credit Score: "))
    age = int(input("Age: "))
    loan = float(input("Loan Amount: "))
    employment = int(input("Employment (1=Yes, 0=No): "))

    income_to_loan = income / loan

    user_data = np.array([[income, credit, age, loan, employment, income_to_loan]])

    result = model.predict(user_data)

    print("\n===== RESULT =====")
    if result[0] == 1:
        print("✅ Loan Approved")
    else:
        print("❌ Loan Rejected")


# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    df = load_data()
    df = preprocess_data(df)

    model, X_test, y_test, X = train_model(df)

    evaluate_model(model, X_test, y_test)
    visualize_data(df, model, X)

    predict_user(model)


# ==========================================
# RUN PROGRAM
# ==========================================
if __name__ == "__main__":
    main()
    
    