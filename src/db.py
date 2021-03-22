from sqlalchemy import create_engine, select, ForeignKey
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker, relationship, backref, mapper
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import Float, Integer, String
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.sql import table, column
from sqlalchemy.orm import load_only
import pandas as pd
from tqdm import tqdm

from database.objects import Sample, SamplePath, SampleClass, SampleSubClass, Features, get_file
from database.tables import create_tables
from sound import features
from database.utils import populate_path_and_classes


import numpy as np



engine = create_engine(
    "postgresql+psycopg2://dev:dev@localhost/data",
    # echo=True
)

# data = pd.read_csv('clean.csv')
# data = data.fillna(0)

# data = data.sample(100)
# data = data.head(5)

# classes_cols = [ col_name for col_name in data.columns if col_name.startswith('class_') ]
# subclasses_cols = [col_name for col_name in data.columns if col_name.startswith('subclass_') ]
# print(classes_cols)
# print(subclasses_cols)

# df_classes = data[data[classes_cols].sum(axis=1) == 1][['path'] + classes_cols]
# df_subclasses = data[data[subclasses_cols].sum(axis=1) == 1][['path'] + subclasses_cols]

# df_classes['class'] = df_classes[classes_cols].idxmax(axis=1).str.replace('class_','')
# df_subclasses['class'] = df_subclasses[subclasses_cols].idxmax(axis=1).str.replace('subclass_','')


create_tables(engine, drop=False)

# with engine.connect() as connection:
#     rows = [{'path': path} for path in data['path']]
#     ins = files_table.insert()
#     connection.execute(ins, *rows)

# def populate_path_and_classes():
#     Session = sessionmaker(bind=engine)
#     session = Session()

#     mfccs = []
#     for i, path in data['path'].items():
#         q = session.query(SamplePath.id).filter(SamplePath.path==path)
#         if not session.query(q.exists()).scalar():
            
#             feat_mfcc = features.fingerprint(path)
#             mfccs.append(feat_mfcc)
#             # print(feat_mfcc.shape)
#             sample = Sample(feat_mfcc)
#             # print(sample.hash)

#             q = session.query(Sample.id).filter(Sample.hash==sample.hash)
#             if session.query(q.exists()).scalar():
#                 oldpath, sample = session.query(SamplePath, Sample).filter(Sample.hash==sample.hash).filter(SamplePath.sample_id == Sample.id).first()
#                 print(f"{oldpath.path} and {path} are identical")
#                 continue
#             sample_path = SamplePath(path)
#             sample.path = sample_path
#             session.add(sample)
#         else:
#             sample = session.query(Sample).join(SamplePath.sample).filter(SamplePath.path==path).one()

#             # print(sample.id)
        
#         if i in df_classes.index:
#             # sample_class = SampleClass(df_classes['class'][i], sample)
#             # sample_class.sample = sample
#             # TODO: test if sample.sample_class exists
#             if sample.sample_class == None:
#                 sample_class = SampleClass(df_classes['class'][i], sample)
#                 session.add(sample_class)
#             else:
#                 sample.sample_class.sample_class = df_classes['class'][i]
#         if i in df_subclasses.index:
#             # sample_subclass = SampleSubClass(df_subclasses['class'][i], sample)
#             # sample_subclass.sample = sample
#             # sample.sample_subclass = sample_subclass
#             if sample.sample_subclass == None:
#                 sample_subclass = SampleSubClass(df_subclasses['class'][i], sample)
#                 session.add(sample_subclass)
#             else:
#                 sample.sample_subclass.sample_subclass = df_subclasses['class'][i]
#         session.merge(sample)
#         # session.merge(sample_class)
#         # session.merge(sample_subclass)
#         session.commit()
#     session.close()

Session = sessionmaker(bind=engine)
session = Session()

populate_path_and_classes('clean.csv', session)

# import simpleaudio
# import numpy as np

# samples = session.query(Sample.mfcc).all()
# for sample, mfcc in zip(samples, mfccs):
#     print('a')
#     array = np.array(sample.mfcc, dtype=np.float32)
#     print(array.dtype)
#     print(array.dtype.itemsize)
#     # print(array)
#     sound = features.fingerprint_to_sound(array)
#     print('done')
#     playback = simpleaudio.play_buffer(
#         sound, 
#         num_channels=1, 
#         bytes_per_sample=sound.dtype.itemsize,
#         sample_rate=16000
#         )
#     print(array.shape)


# for _, item in df_classes.iterrows():
#     # file = session.query(SampleFile).filter(SampleFile.path == item['path']).first()
#     file = get_file(session, path=item['path'])
#     class_name = item['class'].lstrip('class_')
#     # cl = SampleClass(sample_class= class_name)
#     cl = SampleClass(class_name)
#     file.sample_class = cl
#     session.merge(file)
# session.commit()

# for _, item in df_subclasses.iterrows():
#     # file = session.query(SampleFile).filter(SampleFile.path == item['path']).first()
#     file = get_file(session, path=item['path'])
#     class_name = item['class'].lstrip('subclass_')
#     # cl = SampleSubClass(sample_subclass= class_name, file=file)
#     cl = SampleSubClass(class_name, file=file)
#     session.add(cl)
# session.commit()

import sound

# def populate_mfcc():
#     Session = sessionmaker(bind=engine)
#     session = Session()
#     paths = session.query(SamplePath).all()
#     for path in paths:
#         sample = session.query(Sample).filter(Sample.id == path.sample_id).options(load_only("id")).first()
#         mfcc = sound.features.mfcc(path.path)
#         features = session.query(Features).filter(Features.sample_id == path.sample_id).options(load_only("id")).first()
#         # print(features)
#         if features is None:
#             features = Features(mfcc=mfcc)
#         else:
#             features.mfcc = mfcc
#         sample.features = features
#         session.add(sample)
#     session.commit()
#     session.close()

# def populate_yamnet():
#     Session = sessionmaker(bind=engine)
#     session = Session()
#     paths = session.query(SamplePath).all()
#     for path in paths:
#         sample = session.query(Sample).filter(Sample.id == path.sample_id).options(load_only("id")).first()
#         yam = sound.features.yamnet_embedding(path.path)
#         features = session.query(Features).filter(Features.sample_id == path.sample_id).options(load_only("id")).first()
#         # print(features)
#         if features is None:
#             features = Features(yam=yam)
#         else:
#             features.yam = yam
#         sample.features = features
#         session.add(sample)
#     session.commit()
#     session.close()

def populate_feature(key, fun, verbose=True):
    Session = sessionmaker(bind=engine)
    session = Session()
    paths = session.query(SamplePath).all()
    
    if verbose:
        paths = tqdm(paths)

    for path in paths:
        sample = session.query(Sample).filter(Sample.id == path.sample_id).options(load_only("id")).first()
        # feat = {}
        # for key, fun in feature_list.items():
        #     feat[key] = fun(path.path)
        features = session.query(Features).filter(Features.sample_id == path.sample_id).options(load_only("id")).first()
        # print(features)
        if features is None:
            feat = {}
            for key, fun in feature_list.items():
                feat[key] = fun(path.path)
            features = Features(**feat)
        else:
            # for key, fun in feature_list.items():
            q = session.query(Features.id).filter(Features.sample_id == path.sample_id).filter(getattr(Features, key).isnot(None))
            # empty = session.query(Features).filter(Features.sample_id == path.sample_id).filter(getattr(Features, key).isnot(None)).options(load_only("id")).first()
            if not session.query(q.exists()).scalar():
                setattr(features, key, fun(path.path))
            # else:
            #     print('already exist')
        sample.features = features
        session.add(sample)
        session.commit()
    session.commit()
    session.close()


def populate_features(feature_list):
    for key, fun in feature_list.items():
        populate_feature(key, fun)

def populate_mfcc():
    populate_feature('mfcc', sound.features.mfcc)
    # populate_features(feature_list={'mfcc': sound.features.mfcc})

def populate_vgg():
    populate_feature('vgg', sound.features.vggish_embedding)
    # populate_features(feature_list={'vgg': sound.features.vggish_embedding})

def populate_yam():
    populate_feature('yam', sound.features.yamnet_embedding)
    # populate_features(feature_list={'yam': sound.features.yamnet_embedding})


def populate_timbre():
    populate_features(feature_list={
        'hardness': sound.features.hardness,
        'depth': sound.features.depth,
        'brightness': sound.features.brightness,
        'roughness': sound.features.roughness,
        'warmth': sound.features.warmth,
        'sharpness': sound.features.sharpness,
        'boominess': sound.features.boominess,
        })

# populate_features(feature_list={'vgg': sound.features.vggish_embedding, 'mfcc': sound.features.mfcc})
# populate_yam()
# populate_timbre()

from database.utils import FEATURES, fill_feature, fill_features

# def fill_feature(key, fun):
#     Session = sessionmaker(bind=engine)
#     session = Session()
#     sample_ids = set(session.query(SamplePath.sample_id).all())
#     filled_ids = set(session.query(Features.sample_id).filter(getattr(Features, key).isnot(None)).all())
#     ids = sample_ids - filled_ids
#     ids = list(ids)
#     # ids.sort()
#     # ids = ids[:1000]
#     # ids = set(session.query(Features.sample_id).filter(getattr(Features, key) == -1).all())
#     # print(ids)
#     paths = session.query(SamplePath.path).filter(SamplePath.sample_id.in_(ids)).all()
#     features = session.query(Features).filter(Features.sample_id.in_(ids)).options(load_only("id")).all()

#     for path, feature in zip(tqdm(paths), features):
#         setattr(feature, key, fun(path[0]))
#         # print(path[0])
#         # print(getattr(feature, key))
#         session.merge(feature)
#         session.commit()
#     session.close()

# def fill_features(feature_list, session):
#     for key, fun in feature_list.items():
#         fill_feature(key, fun, session)

fill_features(FEATURES, session)

# fill_feature('hardness', sound.features.hardness)

Session = sessionmaker(bind=engine)
session = Session()

fill_features({
    'hardness': sound.features.hardness,
    'depth': sound.features.depth,
    'brightness': sound.features.brightness,
    'roughness': sound.features.roughness,
    'warmth': sound.features.warmth,
    'sharpness': sound.features.sharpness,
    'boominess': sound.features.boominess,
}, session)


features = session.query(Features).first()
print(features.mfcc.shape)
print(features.vgg.shape)
print(features.yam.shape)