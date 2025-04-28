import numpy as np
import pandas as pd
from os import path
import streamlit as st
import pickle
import joblib

st.title("Iris Flower CLassification Application")

filename = "iris_model.pkl"  # pickle file name
model_path = joblib.load(path.join("model", filename))

sepal_length = st.number_input("Insert sepal length")
sepal_width = st.number_input("Insert sepal width")
petal_length = st.number_input("Insert petal length")
petal_width = st.number_input("Insert petal width")

if st.button("Predict"):
    pred = model_path.predict(np.array([[sl, sw, pl, pw]]))
    species_map = {0: "Setosa", 1: "Versicolor", 2: "virginica"}
    st.write("The flower is ", species_map[pred[0]])
