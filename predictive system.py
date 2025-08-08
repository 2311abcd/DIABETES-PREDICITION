# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# loading the save model
loaded_model = pickle.load(open("C:/Users/A-Tech Computers/Downloads/machinelearning/trained_model.sav", 'rb'))
import numpy as np

# Use correct number of features (8)
input_data = (5,166,72,19,175,25.8,0.587,51)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape for single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predict
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

# Output result
if prediction[0] == 0:
    print("PERSON IS NOT DIABETIC")
else:
    print("PERSON IS DIABETIC")

