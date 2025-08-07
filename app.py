import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model
# Ensure the model file name matches the one saved in the previous step
try:
    model = joblib.load('best_XGboosting_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the model training and saving steps first in the notebook.")
    st.stop()

# --- Streamlit Application ---

st.title('Salary Prediction App')

st.write("""
This application predicts data science salaries based on various factors such as experience level, employment type, location, and more. Enter the details below to get a salary prediction.
""")

# Define the features used for training (excluding the target and features not used)
categorical_features = ['work_year','experience_level','job_title','employee_residence', 'employment_type', 'company_location', 'company_size']
numerical_features = ['remote_ratio'] # 'salary' was in X but not used as an input feature for prediction


# Create input fields for the features
work_year = st.selectbox('Work Year', [2020, 2021, 2022])

col1, col2 = st.columns(2)
with col1:
    experience_level = st.selectbox('Experience Level', ['MI', 'SE', 'EN', 'EX'])
with col2:
    employment_type = st.selectbox('Employment Type', ['FT', 'CT', 'PT', 'FL'])

job_title = st.selectbox('Job Title', ['Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer', 'Product Data Analyst', 'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist', 'Business Data Analyst', 'Lead Data Engineer', 'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant', 'BI Data Analyst', 'Research Scientist', 'Machine Learning Manager', 'Data Engineering Manager', 'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist', 'Computer Vision Engineer', 'Principal Data Scientist', 'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher', 'Data Analytics Engineer', 'Applied Data Scientist', 'Director of Data Science', 'Marketing Data Analyst', 'Cloud Data Engineer', 'Computer Vision Software Engineer', 'Director of Data Engineering', 'Data Science Engineer', 'Principal Data Engineer', 'Machine Learning Developer', 'Applied Machine Learning Scientist', 'Data Analytics Manager', 'Head of Data Science', 'Data Specialist', 'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst', 'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer', 'ETL Developer', 'Head of Machine Learning', 'NLP Engineer', 'Lead Machine Learning Engineer', 'Financial Data Analyst'])
employee_residence = st.selectbox('Employee Residence', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
remote_ratio = st.selectbox('Remote Ratio', [0, 50, 100])

col3, col4 = st.columns(2)
with col3:
    company_location = st.selectbox('Company Location', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
with col4:
    company_size = st.selectbox('Company Size', ['S', 'M', 'L'])


# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[work_year, experience_level, employment_type, job_title, employee_residence, remote_ratio, company_location, company_size]],
                              columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

    # --- Preprocessing within the app ---
    # In a real application, it's best to save the fitted preprocessor
    # or at least the fitted individual transformers (StandardScaler, OneHotEncoder)
    # from the training step and load them here.
    # For demonstration purposes due to the loading issue, we will refit
    # the transformers on a dummy DataFrame with the structure of the training data.
    # This is NOT ideal for production as it might not capture all categories
    # or the correct scaling from the original training data.

    # Create a dummy DataFrame with the same columns and structure as the training data
    # This is a simplified representation; ideally, load the original data structure.
    dummy_data = pd.DataFrame(columns=categorical_features + numerical_features)
    # Append the input data to the dummy data for fitting purposes
    dummy_data = dummy_data.append(input_data, ignore_index=True)


    # Recreate and fit the preprocessor (less ideal than loading fitted preprocessor)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like salary, currency if present in dummy)
    )

    # Fit the preprocessor on the dummy data (structure) and then transform the input data
    # This is a workaround due to the loading issue; loading the fitted preprocessor is preferred.
    preprocessor.fit(dummy_data)
    input_processed = preprocessor.transform(input_data)

    # --- End Preprocessing within the app ---


    # Make prediction
    predicted_salary = model.predict(input_processed)

    st.subheader(f'Predicted Salary in USD: ${predicted_salary[0]:,.2f}')
