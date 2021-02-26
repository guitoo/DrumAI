from typing import Optional
from fastapi.templating import Jinja2Templates
from sound import features

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Request
from tensorflow import keras

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

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
    vgg_emb = features.vggish_embedding(samplefile.file)
    result = model.predict(vgg_emb.reshape(1,128))

    class_ = class_names[result.argmax()]

    return {
        "result": str(result),
        "class": class_,
        "classes": class_names
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)

    