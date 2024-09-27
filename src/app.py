import streamlit as st
import pandas as pd
from data import transform_text
from model import predict

# Load training data
train = pd.read_csv('../data/train.csv')

# Streamlit app
st.title("Disaster Prediction from Tweets")
input_text = st.text_area("Enter your tweet text here:")

# Check if input is provided
if st.button("Predict"):
    if input_text:
        X, y, transformed_input = transform_text(train, input_text)
        prediction = predict(X, y, transformed_input)
        
        if prediction[0] == 1:
            st.success("This tweet indicates a disaster!")
        else:
            st.success("This tweet does not indicate a disaster.")
    else:
        st.warning("Please enter some text to predict.")
