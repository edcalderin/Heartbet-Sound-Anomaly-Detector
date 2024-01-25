import requests
import io

BASE_URL = 'http://localhost:8080/predictions/hearbet_model'
filename = 'unzipped_data/set_a/artifact__201012172012.wav'

prediction = requests.post(BASE_URL, data=filename).json()
print(prediction)