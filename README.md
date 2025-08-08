# Data Science Salary Prediction

This project aims to build a robust model to predict data science salaries based on various factors such as experience level, employment type, location, and more.

## Project Overview

The project follows a standard machine learning workflow:

1.  **Data Loading**: Load the dataset containing data science salaries and related features.
2.  **Exploratory Data Analysis (EDA)**: Understand the data distribution, identify potential outliers, and explore relationships between features and salary through visualizations and statistical summaries.
3.  **Data Preprocessing**: Prepare the data for modeling by handling missing values, encoding categorical features, scaling numerical features, and splitting the data into training and testing sets.
4.  **Model Selection and Training**: Choose appropriate regression models and train them on the prepared training data.
5.  **Model Evaluation**: Evaluate the performance of the trained models using metrics like R-squared, Mean Absolute Error, and Mean Squared Error on the testing data.
6.  **Model Prediction**: Use the best-performing model to make salary predictions.
7.  **Deployment (Streamlit App)**: Create a simple web application using Streamlit to allow users to input features and get a predicted salary.

## Dataset

The dataset used in this project is sourced from [mention the source if possible, e.g., Kaggle, a specific file name]. It contains information about data science jobs, including:

*   `work_year`: The year the salary was paid.
*   `experience_level`: The experience level of the employee (e.g., Entry-level, Mid-level, Senior, Executive).
*   `employment_type`: The type of employment (e.g., Full-time, Part-time).
*   `job_title`: The specific job role.
*   `salary`: The salary in the local currency.
*   `salary_currency`: The currency of the salary.
*   `salary_in_usd`: The salary converted to USD.
*   `employee_residence`: The country of residence of the employee.
*   `remote_ratio`: The percentage of time spent working remotely.
*   `company_location`: The country of the company's headquarters.
*   `company_size`: The size of the company (Small, Medium, Large).

## Project Structure

The project is structured as follows:

*   `[Notebook Name].ipynb`: The Jupyter Notebook containing the data loading, EDA, preprocessing, model training, and evaluation steps.
*   `app.py`: The Python script for the Streamlit web application.
*   `requirements.txt`: A file listing the necessary Python packages.
*   `best_XGboosting_model.pkl`: The saved trained machine learning model.
*   `preprocessor.pkl` (Optional but recommended): The saved fitted data preprocessor.

## Getting Started

### Abbreaviations
* `Experience`
      EN: Entry-level / Junior
       MI: Mid-level / Intermediate
       SE: Senior-level / Expert
       EX: Executive-level / Director
* `employment_type`
       PT: Part-time
       FT: Full-time
       CT: Contract
       FL: Freelance
* `remote_ratio`
      0: No remote work (less than 20%)
     50: Partially remote/hybird
    100: Fully remote (more than 80%)
* `company_size`
      S: less than 50 employees (small)
      M: 50 to 250 employees (medium)
      L: more than 250 employees (large)
