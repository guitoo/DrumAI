from typing import Optional
from fastapi.templating import Jinja2Templates
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from sound import features

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.objects import SampleClass

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Request
from tensorflow import keras
from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

engine = create_engine(
    "postgresql+psycopg2://dev:dev@localhost/data",
    # echo=True
)

class_names=[
    'Snare',
    'Kick',
    'Hat',
    'Tom',
    'Cymbal',
    'Clap',
    'Cowbell',
    'Conga',
    'Shaken'
]

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

model = keras.models.load_model('./models/model01')


@app.get("/form")
def form(request: Request):
    return templates.TemplateResponse('form.html', context={'request': request})


# @app.post("/predict")
# async def batch_predict(file: bytes = File(...)):

@app.post("/predict")
async def predict(samplefile: UploadFile = File(...)):
    global model
    logger.info(samplefile.file)

    try:
        suffix = Path(samplefile.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(samplefile.file, tmp)
                tmp_path = Path(tmp.name)
        vgg_emb = features.vggish_embedding(tmp_path)
        tmp_path.unlink()

        result = model.predict(vgg_emb.reshape(1,128))



        class_ = class_names[result.argmax()]

        return {
            "result": str(result),
            "class": class_,
            "classes": class_names
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)

    