from typing import Tuple
import streamlit as st
import plotly.express as px
import os
import requests

st.set_page_config(layout='wide')

endpoint_url: str = os.getenv('ENDPOINT_URL', 'http://localhost:8000')

def process_data(file) -> Tuple:
    file_request = {'file': file}
    response = requests.post(f'{endpoint_url}/predict', files=file_request).json()
    return response.get('normal'), response.get('abnormal')

def main():
    st.title('Heartbeat Anomaly Detector')

    left_column, right_column = st.columns(2)

    with left_column:
        uploaded_file = st.file_uploader("Choose a file", type=["wav"])

        if uploaded_file is not None:
            st.success("File successfully uploaded!")

        if st.button('Predict'):
            with st.spinner('Processing...'):
                normal_value, abnormal_value = process_data(uploaded_file)

            st.text('Prediction complete!')

    with right_column:
        if 'normal_value' in locals() and 'abnormal_value' in locals():
            fig = px.bar(x = ['Normal', 'Abnormal'], 
                         y = [normal_value, abnormal_value],
                         labels = {'x': 'Anomaly', 'y': 'Probability'},
                         title = 'Heartbeat Predictions')
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
