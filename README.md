import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def customer_churn_prediction():
    st.title("Customer Churn Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Churn' not in df.columns:
            st.error("CSV file must contain a 'Churn' column.")
            return
        
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        st.write("Model Accuracy:", model.score(X_test, y_test))

if __name__ == "__main__":
    customer_churn_prediction()
