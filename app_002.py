import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('modell_001.pkl', 'rb'))

st.title("HR Employee Exit Prediction App")
st.write("Predict whether an employee will leave the company.")

# User Inputs
satisfaction_level = st.slider("Satisfaction Level(%)", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score(%)", 0.0, 1.0, 0.6)
number_project = st.slider("Number of Projects", 1, 10, 3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
time_spend_company = st.slider("Years at Company", 0, 15, 3)

work_accident_input = st.selectbox("Had Work Accident?", ["No", "Yes"])
Work_accident = 1 if work_accident_input == "Yes" else 0


promotion_input = st.selectbox("Got Promotion in Last 5 Years?", ["No", "Yes"])
promotion_last_5years = 1 if promotion_input == "Yes" else 0

Department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

# Combine into dataframe
input_data = pd.DataFrame([{
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    'Department': Department,
    'salary': salary
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Will Leave the Company" if prediction == 1 else "Will Stay in the Company"
    st.success(f"Prediction: {result}")
