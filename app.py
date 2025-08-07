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
    model = joblib.load('best_gradient_boosting_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the model training and saving steps first.")
    st.stop()


# Assuming you have the preprocessor fitted on the training data
# You would ideally save and load the preprocessor as well
# For this example, we will recreate a similar preprocessor structure
# based on the columns used during training.
# In a real application, save and load the fitted preprocessor.

# Identify categorical and numerical features (should match the training preprocessing)
categorical_features = ['work_year','experience_level', 'employment_type', 'company_location', 'company_size']
numerical_features = ['remote_ratio', 'salary'] # Including 'salary' here because it was in the original X, although not used for prediction input

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer (should match the training preprocessing)
# Drop the columns that are not used as features or target variable
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop columns that are not in numerical_features or categorical_features
)


# --- Streamlit Application ---

st.title('Salary Prediction App')

st.write("""
Enter the details below to predict the salary.
""")

# Create input fields for the features
# These should match the features used for training the model (excluding 'salary_in_usd' and 'salary')
# We need to get the unique values for categorical features from the original data or a saved list
# For simplicity, using some example values here. In a real app, load these from saved data.
work_year = st.selectbox('Work Year', [2020, 2021, 2022])
experience_level = st.selectbox('Experience Level', ['MI', 'SE', 'EN', 'EX'])
employment_type = st.selectbox('Employment Type', ['FT', 'CT', 'PT', 'FL'])
remote_ratio = st.selectbox('Remote Ratio', [0, 50, 100])
company_size = st.selectbox('Company Size', ['S', 'M', 'L'])
company_location = st.text_input('Company Location (e.g., US)') # Using text input for simplicity


# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[work_year, experience_level, employment_type, "dummy_job_title", 0, "dummy_currency", "dummy_residence", remote_ratio, company_location, company_size]],
                              columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

    # Apply the same preprocessing as used during training
    # Note: This requires fitting the preprocessor again which is not ideal.
    # A better approach is to save and load the fitted preprocessor object.
    # For demonstration purposes, we'll fit it on the original data structure (excluding target)
    # to ensure the columns align for the one-hot encoding.
    # This part needs to be robust in a real application.
    # Let's create a dummy preprocessor just for the structure
    dummy_X = pd.DataFrame(columns=['work_year','experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])
    dummy_X = dummy_X.append(input_data, ignore_index=True)

    # Fit the preprocessor on dummy data to get the structure (not ideal)
    # In a real app, load the fitted preprocessor
    preprocessor.fit(dummy_X) # This is a simplification, load the fitted one instead

    input_processed = preprocessor.transform(input_data)


    # Make prediction
    predicted_salary = model.predict(input_processed)

    st.subheader(f'Predicted Salary in USD: ${predicted_salary[0]:,.2f}')
