# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:22:32 2025

@author: A-Tech Computers
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/A-Tech Computers/Downloads/machinelearning/trained_model.sav", 'rb'))

# Creating a function for prediction 
def diabetes_prediction(input_data):
    
    
    

    # Convert to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    # Output result
    if (prediction[0] == 0):
        return 'THE PERSON IS NOT DIABETIC'
    else:
        return 'THE PERSON IS DIABETIC'
    
    
def main():
    
    # giving the title
    st.title('DIABETES PREDICTION WEB APP')
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input(' Age of the Person')
    
    # code for prediciton 
    diagnosis = ''
    
    # creating a Button for Prediciton 
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':

    main()
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    