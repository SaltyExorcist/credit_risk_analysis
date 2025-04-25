import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_data, encode_and_scale, generate_risk_label


def main():
    model = joblib.load('credit_risk_model.pkl')
    
    st.set_page_config(page_title="Credit Risk App")
    st.title("Credit Risk Prediction & Explanation")

    df = load_data('german_credit_data.csv')
    original_df = df.copy()

    with st.sidebar.form('input_form'):
        st.header("Applicant Details")
        Age = st.slider('Age', 18, 75, 30)
        Credit_amount = st.number_input('Credit amount', min_value=250.0, max_value=20000.0, value=1000.0)
        Duration = st.slider('Duration (months)', 4, 72, 12)

        Sex = st.selectbox('Sex', original_df['Sex'].unique())
        Job = st.selectbox('Job', original_df['Job'].unique())
        Housing = st.selectbox('Housing', original_df['Housing'].unique())
        Saving_accounts = st.selectbox('Saving accounts', original_df['Saving accounts'].fillna('unknown').unique())
        Checking_account = st.selectbox('Checking account', original_df['Checking account'].fillna('unknown').unique())
        Purpose = st.selectbox('Purpose', original_df['Purpose'].unique())

        submitted = st.form_submit_button('Predict')

    if submitted:
        input_data = pd.DataFrame([{
            'Age': Age,
            'Credit amount': Credit_amount,
            'Duration': Duration,
            'Sex': Sex,
            'Job': Job,
            'Housing': Housing,
            'Saving accounts': Saving_accounts,
            'Checking account': Checking_account,
            'Purpose': Purpose
        }])

        combined = pd.concat([input_data, original_df.iloc[:1]], axis=0)  # ensure all categories exist
        combined, _, _ = encode_and_scale(combined)
        input_encoded = combined.iloc[[0]]

        # Align with model's expected feature order
        input_encoded = input_encoded[model.feature_names_in_]

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0, 1]

        st.subheader("Prediction Result")
        st.write("**Bad Credit Risk**" if prediction == 1 else "**Good Credit Risk**")
        st.write(f"Probability of Bad Credit Risk: {probability:.2f}")

       

if __name__ == "__main__":
    main()

