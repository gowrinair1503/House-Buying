#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving the model

# Define column names
column_names = ['Age', 'Income', 'Spending_Score', 'Debt', 'Credit_Score',
                'Shopping_Frequency', 'Social_Media_Usage', 'Online_Shopping',
                'Customer_Loyalty', 'Promotion_Response']

# Generating a synthetic dataset
X, y = make_classification(n_samples=1000,    # Number of samples
                           n_features=10,     # Number of features
                           n_informative=8,   # Number of informative features
                           n_redundant=2,     # Number of redundant features
                           n_classes=2,       # Number of classes
                           random_state=42)   # Random state for reproducibility

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'model5.pkl')


# In[6]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib

# Load the trained model
model = joblib.load('C:\\Users\\admin\\Downloads\\model5.pkl')  # Update with your actual model file path

def main():
    st.title('Random Forest Classifier Demo')

    # Add description or instructions
    st.write('This app demonstrates a Random Forest Classifier.')

    # Add sidebar with options
    st.sidebar.title('Options')
    action = st.sidebar.selectbox('Select Action', ['Home', 'Predict'])

    if action == 'Home':
        st.write('This is the home page.')

    elif action == 'Predict':
        st.subheader('Make a Prediction')
        st.write('Enter values for the following features:')

        # Create input fields for user to enter feature values
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        income = st.number_input('Income', min_value=0, max_value=200000, value=50000)
        spending_score = st.slider('Spending Score', min_value=1, max_value=100, value=50)
        debt = st.number_input('Debt', min_value=0, max_value=100000, value=1000)
        credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=600)
        shopping_frequency = st.number_input('Shopping Frequency', min_value=0, max_value=10, value=5)
        social_media_usage = st.slider('Social Media Usage', min_value=0, max_value=24, value=5)
        online_shopping = st.radio('Online Shopping', ['Yes', 'No'])
        customer_loyalty = st.radio('Customer Loyalty', ['Yes', 'No'])
        promotion_response = st.radio('Promotion Response', ['Yes', 'No'])

        # Convert categorical inputs to numerical
        online_shopping = 1 if online_shopping == 'Yes' else 0
        customer_loyalty = 1 if customer_loyalty == 'Yes' else 0
        promotion_response = 1 if promotion_response == 'Yes' else 0

        # Create a numpy array with the input data
        input_data = np.array([[age, income, spending_score, debt, credit_score,
                                shopping_frequency, social_media_usage, online_shopping,
                                customer_loyalty, promotion_response]])

        # Predict the output using the loaded model
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display the prediction and probability
        st.subheader('Prediction')
        if prediction[0] == 0:
            st.write('Customer is predicted as **Not Likely** to respond to promotion.')
        else:
            st.write('Customer is predicted as **Likely** to respond to promotion.')

        st.subheader('Prediction Probability')
        st.write(f'Probability of Not Likely: {prediction_proba[0][0]:.2f}')
        st.write(f'Probability of Likely: {prediction_proba[0][1]:.2f}')

if __name__ == '__main__':
    main()


# In[ ]:




