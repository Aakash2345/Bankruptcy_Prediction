import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open("E://Models//Bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Bankruptcy Prevention")

# Define value options for each column
risk_values = [0.0, 0.5, 1.0]

tab0, tab1 = st.tabs(["Predict Bankruptcy", "Visualize feature Importance"])

with tab0:
    # Individual input dropdowns for each feature
    industrial_risk = st.selectbox("Industrial Risk", risk_values)
    management_risk = st.selectbox("Management Risk", risk_values)
    financial_flexibility = st.selectbox("Financial Flexibility", risk_values)
    credibility = st.selectbox("Credibility", risk_values)
    competitiveness = st.selectbox("Competitiveness", risk_values)
    operating_risk = st.selectbox("Operating Risk", risk_values)

    # When user submits the input
    if st.button("Predict"):
        input_data = {
        "industrial_risk": industrial_risk,
        "management_risk": management_risk,
        "financial_flexibility": financial_flexibility,
        "credibility": credibility,
        "competitiveness": competitiveness,
        "operating_risk": operating_risk
        }

        # Convert to 1-row DataFrame with correct column names
        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)
        if result == 0:
            st.warning("Bankruptcy Risk")
        else:
            st.success('Non-Bankruptcy')
with tab1:
    features = ['industrial_risk', 'management_risk', 'financial_flexibility',
            'credibility', 'competitiveness', 'operating_risk']

    coefs = model.coef_[0]
    
    colors = ['green' if c > 0 else 'red' for c in coefs]

    fig, ax = plt.subplots()
    ax.bar(features, coefs, color = colors)
    ax.set_title("Feature Impact (Model Coefficients)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient Value")
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    st.write('Positive coefficient → pushes toward Non-Bankruptcy')
    st.write('Negative coefficient → pushes toward Bankruptcy')