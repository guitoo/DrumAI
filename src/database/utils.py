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
from sound.path import get_full_path


import numpy as np

FEATURES = {
    'hardness': features.hardness,
    'depth': features.depth,
    'brightness': features.brightness,
    'roughness': features.roughness,
    'warmth': features.warmth,
    'sharpness': features.sharpness,
    'boominess': features.boominess,
    'contrast': features.contrast,
    'zero_crossing_rate': features.zero_crossing_rate,
    'spectral_flatness': features.spectral_flatness
}

def fill_feature(key, fun, session):
    sample_ids = set(session.query(SamplePath.sample_id).all())
    filled_ids = set(session.query(Features.sample_id).filter(getattr(Features, key).isnot(None)).all())
    ids = sample_ids - filled_ids
    ids = list(ids)

    q = session.query(SamplePath.path, Features)
    q = q.filter(SamplePath.sample_id == Features.sample_id)
    q = q.filter(Features.sample_id.in_(ids))
    q = q.options(load_only(Features.id))
    paths_and_features = q.all()

    # print (paths_and_features)

    for path, feature in tqdm(paths_and_features):
        # print(path)
        # print(feature.id)
    # for path, feature in zip(tqdm(paths), features):
        setattr(feature, key, fun(get_full_path(path)))
        # print(path[0])
        # print(getattr(feature, key))
        session.merge(feature)
        session.commit()

def fill_features(feature_list, session):
    for key, fun in feature_list.items():
        print(f'filling feature {key}')
        fill_feature(key, fun, session)


def populate_path_and_classes(filename, session):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    classes_cols = [ col_name for col_name in data.columns if col_name.startswith('class_') ]
    subclasses_cols = [col_name for col_name in data.columns if col_name.startswith('subclass_') ]
    df_classes = data[data[classes_cols].sum(axis=1) == 1][['path'] + classes_cols]
    df_subclasses = data[data[subclasses_cols].sum(axis=1) == 1][['path'] + subclasses_cols]
    df_classes['class'] = df_classes[classes_cols].idxmax(axis=1).str.replace('class_','')
    df_subclasses['class'] = df_subclasses[subclasses_cols].idxmax(axis=1).str.replace('subclass_','')
    # classes_cols = [ col_name for col_name in data.columns if col_name.startswith('class_') ]
    # subclasses_cols = [col_name for col_name in data.columns if col_name.startswith('subclass_') ]

    # df_classes = data[data[classes_cols].sum(axis=1) == 1][['path'] + classes_cols]
    # df_subclasses = data[data[subclasses_cols].sum(axis=1) == 1][['path'] + subclasses_cols]

    # df_classes['class'] = df_classes[classes_cols].idxmax(axis=1)
    # df_subclasses['class'] = df_subclasses[subclasses_cols].idxmax(axis=1)

    # # mfccs = []
    # for i, path in data['path'].items():
    #     q = session.query(SamplePath.id).filter(SamplePath.path==path)
    #     if session.query(q.exists()).scalar():
    #         continue
    #     feat_mfcc = features.fingerprint(get_full_path(path))
    #     # mfccs.append(feat_mfcc)
    #     # print(feat_mfcc.shape)
    #     sample = Sample(feat_mfcc)
    #     # print(sample.hash)

    #     q = session.query(Sample.id).filter(Sample.hash==sample.hash)
    #     if session.query(q.exists()).scalar():
    #         oldpath, sample = session.query(SamplePath, Sample).filter(Sample.hash==sample.hash).filter(SamplePath.sample_id == Sample.id).first()
    #         print(f"{oldpath.path} and {path} are identical")
    #         continue

    #     sample_path = SamplePath(path)
    #     sample.path = sample_path
    #     if i in df_classes.index:
    #         sample_class = SampleClass(df_classes['class'][i].lstrip('class_'))
    #         sample.sample_class = sample_class
    #     if i in df_subclasses.index:
    #         sample_subclass = SampleSubClass(df_subclasses['class'][i].lstrip('subclass_'))
    #         sample.sample_subclass = sample_subclass
    #     session.merge(sample)
    # session.commit()
    mfccs = []
    for i, path in data['path'].items():
        q = session.query(SamplePath.id).filter(SamplePath.path==path)
        if not session.query(q.exists()).scalar():
            feat_mfcc = features.fingerprint(path)
            mfccs.append(feat_mfcc)
            sample = Sample(feat_mfcc)
            q = session.query(Sample.id).filter(Sample.hash==sample.hash)
            if session.query(q.exists()).scalar():
                oldpath, sample = session.query(SamplePath, Sample).filter(Sample.hash==sample.hash).filter(SamplePath.sample_id == Sample.id).first()
                print(f"{oldpath.path} and {path} are identical")
                continue
            sample_path = SamplePath(path)
            sample.path = sample_path
            session.add(sample)
        else:
            sample = session.query(Sample).join(SamplePath.sample).filter(SamplePath.path==path).one()        
        if i in df_classes.index:
            if sample.sample_class == None:
                sample_class = SampleClass(df_classes['class'][i], sample)
                session.add(sample_class)
            else:
                sample.sample_class.sample_class = df_classes['class'][i]
        if i in df_subclasses.index:
            if sample.sample_subclass == None:
                sample_subclass = SampleSubClass(df_subclasses['class'][i], sample)
                session.add(sample_subclass)
            else:
                sample.sample_subclass.sample_subclass = df_subclasses['class'][i]
        session.merge(sample)
        session.commit()
    session.close()
