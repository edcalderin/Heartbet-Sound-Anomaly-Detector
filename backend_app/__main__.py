from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from backend_app.model_handler import ModelHandler
from backend_app.utils import Utils
import uvicorn
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s')

resources = dict()

@asynccontextmanager
async def app_life_span(_: FastAPI):
    logging.info('Loading model...')
    resources['model'] = Utils.load_model()
    yield
    resources.clear()

app = FastAPI(lifespan = app_life_span)

@app.post('/predict')
def predict(file: UploadFile):
    model_handler = ModelHandler(model = resources['model'], file_content = file.file)
    return model_handler()

if __name__ == '__main__':
    uvicorn.run(app)
