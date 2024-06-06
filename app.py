import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler objects
with open('final_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to scale features and predict using the classifier
def predict_cluster(input_features):
    # Convert input data into a numpy array
    input_features = np.array(input_features).reshape(1, -1)

    # Scale the features using the pre-fitted scaler
    scaled_features = scaler.transform(input_features)

    # Predict using the classifier
    prediction = classifier.predict(scaled_features)

    return prediction

# Main function for the Streamlit app
def main():
    st.title("Customer Segmentation App")

    # Create input fields for each feature
    education = st.selectbox("Education", [0, 1, 2])
    marital_status = st.selectbox("Marital Status", [0, 1])
    income = st.number_input("Income", min_value=0, max_value=1000000,step=1000)
    recency = st.number_input("Recency", min_value=0, max_value=100, step=1)
    num_web_visits_month = st.number_input("Number of Web Visits perMonth", min_value=0, max_value=30, step=1)
    complain = st.selectbox("Complain", [0, 1])
    age = st.number_input("Age", min_value=0, max_value=150, step=1)
    total_accepted_cmp = st.number_input("Total Accepted Campaigns",min_value=0, max_value=10, step=1)
    total_purchases = st.number_input("Total Purchases", min_value=0,max_value=50, step=1)
    children = st.number_input("Children", min_value=0, max_value=10, step=1)
    spending = st.number_input("Spending", min_value=0,max_value=1000000, step=100)
    savings = st.selectbox("Savings", [0, 1])

    # When the user clicks the 'Predict' button, make a prediction
    if st.button("Predict"):
        input_features = [
            education, marital_status, income, recency, num_web_visits_month,
            complain, age, total_accepted_cmp, total_purchases, children,
            spending, savings ]
        prediction = predict_cluster(input_features)
        st.write(f"The predicted cluster is: {prediction[0]}")

if __name__ == '__main__':
    main()