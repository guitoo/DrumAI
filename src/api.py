from typing import Optional
from fastapi.templating import Jinja2Templates
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from sound import features

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.objects import SampleClass, Sample, Features, SamplePath ,UserClassVote, User
from database.utils import FEATURES, fill_user_features

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Request, Body
from tensorflow import keras
from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path

import logging

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

engine = create_engine(
    "postgresql+psycopg2://dev:dev@postgres/data",
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

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

model = keras.models.load_model('./models/model01')


# @app.get("/form")
# def form(request: Request):
#     return templates.TemplateResponse('form.html', context={'request': request})


# @app.post("/predict")
# async def batch_predict(file: bytes = File(...)):


# global logger
# @app.on_event("startup")
# async def startup_event():
# logger = logging.getLogger("uvicorn.access")
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logger.addHandler(handler)

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


@app.post("/correct")
async def correct_prediction(username: str = Body(...), class_: str = Body(...), samplefile: UploadFile = File(...)):
    global model
    logger.warning(samplefile.filename)
    Session = sessionmaker(bind=engine)
    session = Session()
    suffix = Path(samplefile.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(samplefile.file, tmp)
        tmp_path = Path(tmp.name)
        
    try:
        

        feat_mfcc = features.fingerprint(tmp_path)
        sample = Sample(feat_mfcc)

        q = session.query(Sample.id).filter(Sample.hash==sample.hash)
        if session.query(q.exists()).scalar():
            sample = session.query(Sample).filter(Sample.hash==sample.hash).first()
        else:
            print('adding sample')
            session.add(sample)
            sample = session.query(Sample).filter(Sample.hash==sample.hash).first()
        
        fill_user_features(FEATURES, sample, tmp_path, session)
        logger.warning('features Done')

        logger.warning(f"sample hash:{sample.hash}")
        

        logger.warning(f"sample id:{sample.id}")

        # if sample.sample_class == None:
        #     sample.sample_class = SampleClass(class_, sample)
        #     session.add(sample)
        # else:
        #     sample.sample_class.sample.sample_class = class_

        

        user = session.query(User).filter(User.name == username).first()
        if user is None:
            user = User(name=username)
            session.add(user)
            session.flush()
            session.refresh(user)
        logger.warning(user.id)
        vote = session.query(UserClassVote).filter(UserClassVote.user_id == user.id).filter(UserClassVote.sample_id == sample.id).first()
        if vote is None:
            vote = UserClassVote(user_id=user.id, sample_id=sample.id, sample_class=class_)
        else:
            vote.sample_class = class_
        session.add(vote)

        session.commit()
        # vote = UserClassVote()

        # result = model.predict(vgg_emb.reshape(1,128))



        # class_ = class_names[result.argmax()]

        return {
            'message': 'success'
        }
    except Exception as e:
        session.rollback()
        return JSONResponse(
            status_code=400,
            content={"message": str(e)},
        )
    finally:
        tmp_path.unlink()

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True, debug=True)

    