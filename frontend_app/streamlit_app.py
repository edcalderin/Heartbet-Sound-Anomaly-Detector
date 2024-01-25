from typing import Tuple
import streamlit as st
import numpy as np
import plotly.express as px
import os
import requests
import time

base_url: str = os.getenv('BACKEND_APP_URL', 'http://localhost:8000')

def process_data(file) -> Tuple:
    response = requests.post(base_url, data=file).json()

    return response.get('normal'), response.get('abnormal')

def main():
    st.title('Streamlit Application')

    uploaded_file = st.file_uploader("Choose a file", type=["wav"])

    if uploaded_file is not None:
        st.success("File successfully uploaded!")

        if st.button('Predict'):
            st.text('Processing...')

            normal_value, abnormal_value = process_data(uploaded_file)

            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                progress_bar.progress(percent_complete + 1)

            st.text('Prediction complete!')

            fig = px.bar(x=['Normal', 'Abdormal'], y=[normal_value, abnormal_value], labels={'y': 'Values'}, title='Hearbet predictions')
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
