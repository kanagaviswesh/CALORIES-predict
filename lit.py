import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('pickle.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict Calories using the loaded model
def predict_Calories(Total_Fat,Sodium,Sugars,Protein):
    features = np.array([Total_Fat,Sodium,Sugars,Protein]) # type: ignore
    features = features.reshape(1,-1)
    Calories = model.predict(features)
    return Calories[0]

# Streamlit UI
st.title('Calories Prediction')
st.write("""
## Input Features
Enter the values for the input features to predict Calories.
""")

# Input fields for user 
Total_Fat = st.number_input('Total Fat (g)') # type: ignore
Sodium = st.number_input('Sodium (mg)') # type: ignore
Sugars = st.number_input('Sugars (g)') # type: ignore
Protein = st.number_input('Protein_(g)') # type: ignore


# Prediction button
if st.button('Predict'):
    # Predict Calories
    Calories_prediction = predict_Calories(Total_Fat,Sodium,Sugars,Protein) # type: ignore
    st.write(f"Predicted Calories: {Calories_prediction}")