import streamlit as st
import numpy as np
import joblib
import os

MODEL_DIR = "Machinelearning"
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
svm_linear = joblib.load(os.path.join(MODEL_DIR, "svm_linear_model.pkl"))

st.title("Hogwarts House Prediction")

bravery = st.slider("Bravery", min_value=0, max_value=10, step=1)
intelligence = st.slider("Intelligence", min_value=0, max_value=10, step=1)
ambition = st.slider("Ambition", min_value=0, max_value=10, step=1)
dark_arts_knowledge = st.slider("Dark Arts Knowledge", min_value=0, max_value=10, step=1)
creativity = st.slider("Creativity", min_value=0, max_value=10, step=1)

model_choice = st.selectbox("Choose the model", ["KNN", "SVM"])

if st.button("Predict"):
    new_data = np.array([[bravery, intelligence, ambition, dark_arts_knowledge, creativity]])

    if model_choice == "KNN":
        predicted_house = knn.predict(new_data)
    else: 
        predicted_house = svm_linear.predict(new_data)

    house_dict = {1: 'Gryffindor', 2: 'Ravenclaw', 3: 'Hufflepuff', 4: 'Slytherin'}
    st.success(f"The predicted House is: **{house_dict[predicted_house[0]]}**")
