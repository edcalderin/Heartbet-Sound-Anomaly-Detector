import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
import os

base_url = os.getenv('BASE_URL')

def predict(file_path):
    # Replace this with your actual prediction logic
    # Here, we're generating random values for demonstration purposes
    response = requests.post(base_url).json()
    normal_height = np.random.uniform(0, 1)
    abnormal_height = np.random.uniform(0, 1)
    return normal_height, abnormal_height

# Streamlit App
def main():
    st.title("Sound File Analysis")

    # File uploader for sound file
    uploaded_file = st.file_uploader("Select a sound file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav', start_time=0)

        # Button to trigger prediction
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                normal_height, abnormal_height = predict(uploaded_file)

            # Bar plot to display prediction results
            results = pd.DataFrame({
                'Category': ['Normal', 'Abnormal'],
                'Height': [normal_height, abnormal_height]
            })

            st.bar_chart(results.set_index('Category'))

if __name__ == "__main__":
    main()
