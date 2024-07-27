import streamlit as st
import requests

st.set_page_config(layout="wide")

st.title('Bank Customer Churn Prediction')

# API URL
url = 'http://localhost:8000/predict/'

# Split the screen into two columns
col1, col2 = st.columns(2)

# User inputs in the first column
with col1:
    credit_score = st.number_input('Credit Score', min_value=0)
    country = st.selectbox('Country', options=['France', 'Spain', 'Germany'])
    gender = st.selectbox('Gender', options=['Female', 'Male'])
    age = st.number_input('Age', min_value=18, max_value=100)
    tenure = st.number_input('Tenure', min_value=0, max_value=20)

with col2:
    balance = st.number_input('Balance', min_value=0.0, format="%.2f")
    products_number = st.number_input('Number of Products', min_value=0, max_value=10)
    credit_card = st.checkbox('Has Credit Card')
    active_member = st.checkbox('Active Member')
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
    model_type = st.selectbox('Model Type', options=['Logistic Regression', 'Random Forest', 'SVM'])

    # Prepare data
    data = {
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary,
        "model_type": model_type
    }

    # Submit button
    if st.button('Predict'):
        # Send request to API
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            prediction = response.json()
            result = 'Churn' if prediction['prediction'][0] == 1 else 'No Churn'
        else:
            result = f"Failed to get prediction, server returned status code {response.status_code}"
        
        # Display the result below the button
        st.write('Prediction:', result)
