import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model and preprocessor
try:
    model = joblib.load('best_XGboosting_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please run the model training, preprocessing, and saving steps first in the notebook.")
    st.stop()

# --- Streamlit Application ---

st.title('Salary Prediction App')

st.write("""
This application predicts data science salaries based on various factors such as experience level, employment type, location, and more. Enter the details below to get a salary prediction.
""")

# Create input fields for the features
# These should match the features used for training the model (excluding 'salary_in_usd' and 'salary')
# We need to get the unique values for categorical features from the original data or a saved list
# In a real app, load these from saved data. For simplicity, using example values for now.
work_year = st.selectbox('Work Year', [2020, 2021, 2022])

col1, col2 = st.columns(2)
with col1:
    experience_level = st.selectbox('Experience Level(EN: Entry-level / Junior, MI: Mid-level / Intermediate, SE: Senior-level / Expert, EX: Executive-level / Director)', ['MI', 'SE', 'EN', 'EX'])
with col2:
    employment_type = st.selectbox('Employment Type(PT: Part-time, FT: Full-time, CT: Contract, FL: Freelance)', ['FT', 'CT', 'PT', 'FL'])

job_title = st.selectbox('Job Title', ['Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer', 'Product Data Analyst', 'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist', 'Business Data Analyst', 'Lead Data Engineer', 'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant', 'BI Data Analyst', 'Research Scientist', 'Machine Learning Manager', 'Data Engineering Manager', 'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist', 'Computer Vision Engineer', 'Principal Data Scientist', 'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher', 'Data Analytics Engineer', 'Applied Data Scientist', 'Director of Data Science', 'Marketing Data Analyst', 'Cloud Data Engineer', 'Computer Vision Software Engineer', 'Director of Data Engineering', 'Data Science Engineer', 'Principal Data Engineer', 'Machine Learning Developer', 'Applied Machine Learning Scientist', 'Data Analytics Manager', 'Head of Data Science', 'Data Specialist', 'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst', 'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer', 'ETL Developer', 'Head of Machine Learning', 'NLP Engineer', 'Lead Machine Learning Engineer', 'Financial Data Analyst'])
salary_currency = st.selectbox('Salary Currency', ['EUR', 'USD', 'GBP', 'HUF', 'INR', 'MXN', 'TRY', 'CAD', 'DKK', 'PLN', 'AED', 'JPY', 'CNY', 'SGD', 'CLP', 'BRL', 'NZD', 'PHP'])
employee_residence = st.selectbox('Employee Residence', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
remote_ratio = st.selectbox('Remote Ratio (0: No remote work (less than 20%), 50: Partially remote/hybird, 100: Fully remote (more than 80%))', [0, 50, 100])

col3, col4 = st.columns(2)
with col3:
    company_location = st.selectbox('Company Location', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
with col4:
    company_size = st.selectbox('Company Size(S: less than 50 employees (small), M: 50 to 250 employees (medium), L: more than 250 employees (large))', ['S', 'M', 'L'])


# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[work_year, experience_level, employment_type, job_title, 0, salary_currency, employee_residence, remote_ratio, company_location, company_size]],
                              columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

    # Apply the loaded preprocessor to the input data
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    predicted_salary = model.predict(input_processed)

    st.subheader(f'Predicted Salary in USD: ${predicted_salary[0]:,.2f}')
