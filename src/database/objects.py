from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Float, Integer, String, TypeDecorator
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import validates
from psycopg2.extensions import register_adapter, AsIs

import hashlib
import numpy as np

Base = declarative_base()

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
# def addapt_numpy_array(numpy_array):
#     return AsIs(tuple(numpy_array))
    # return AsIs(numpy_array.tolist())
# register_adapter(np.ndarray, addapt_numpy_array)
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.float32, addapt_numpy_float32)

class Array2D(TypeDecorator):
    impl = postgresql.ARRAY(Float, dimensions=2)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return np.array(value)

class Array1D(TypeDecorator):
    impl = postgresql.ARRAY(Float, dimensions=1)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # print('tolist', type(value))
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # print('tonumpy', type(value))
        # convert sql string to python time
        return np.array(value)


class Sample(Base):
    __tablename__ = 'samples'

    id = Column('id', Integer, primary_key=True)
    hash = Column('hash', String, unique=True)
    fingerprint = Column('fingerprint', postgresql.ARRAY(Float, dimensions=2))
    path = relationship("SamplePath", uselist=False, back_populates="sample")
    sample_class = relationship("SampleClass", uselist=False, back_populates="sample")
    sample_subclass = relationship("SampleSubClass", uselist=False, back_populates="sample")
    features = relationship("Features", uselist=False, back_populates="sample")

    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.hash = hashlib.sha1(fingerprint.view(np.uint8)).hexdigest()
        # print(mfcc.view(np.uint8))


class SamplePath(Base):
    __tablename__ = 'sample_paths'

    id = Column('id', Integer, primary_key=True)
    sample_id = Column('sample_id', Integer, ForeignKey("samples.id"), unique=True)
    path = Column('path', String)
    sample = relationship("Sample", uselist=False, back_populates="path")

    def __init__(self, path, sample=None):
        self.path = path
        if sample is not None:
            self.sample = sample


class SampleClass(Base):
    __tablename__ = 'sample_classes'

    id = Column('id', Integer, primary_key=True)
    sample_id = Column('sample_id', Integer, ForeignKey("samples.id"), unique=True)
    sample_class = Column('sample_class', String)
    sample = relationship("Sample", uselist=False, back_populates="sample_class")

    def __init__(self, sample_class, file=None):
        self.sample_class = sample_class
        if file is not None:
            self.file = file


class SampleSubClass(Base):
    __tablename__ = 'sample_subclasses'

    id = Column('id', Integer, primary_key=True)
    sample_id = Column('sample_id', Integer, ForeignKey("samples.id"), unique=True)
    sample_subclass = Column('sample_subclass', String)
    sample = relationship("Sample", uselist=False, back_populates="sample_subclass")

    def __init__(self, sample_subclass, sample=None):
        self.sample_subclass = sample_subclass
        if sample is not None:
            self.sample = sample


class Features(Base):
    __tablename__ = 'features'

    id = Column('id', Integer, primary_key=True)

    sample_id = Column('sample_id', Integer, ForeignKey("samples.id"), unique=True)
    vgg = Column("vgg", Array1D)
    yam = Column("yam", Array1D)
    mfcc = Column("mfcc", Array2D)
    # vgg = Column("vgg", postgresql.ARRAY(Float, dimensions=1))
    # yam = Column("yam", postgresql.ARRAY(Float, dimensions=1))
    # mfcc = Column("mfcc", postgresql.ARRAY(Float, dimensions=2))
    hardness = Column("hardness", Float)
    depth = Column("depth", Float)
    brightness = Column("brightness", Float)
    roughness = Column("roughness", Float)
    warmth = Column("warmth", Float)
    sharpness = Column("sharpness", Float)
    boominess = Column("boominess", Float)
    sample = relationship("Sample", uselist=False, back_populates="features")

    @validates('mfcc')
    def validate_mfcc(self, key, mfcc):
        # print(mfcc)
        return mfcc.tolist()

    @validates('yam')
    def validate_yam(self, key, yam):
        # print(yam)
        return yam.tolist()

    @validates('vgg')
    def validate_vgg(self, key, vgg):
        # print(vgg)
        return vgg.tolist()



def get_file(session, path=None, sample_id=None):
    if path is not None:
        return session.query(Sample).filter(Sample.path == path).first()
    if sample_id is not None:
        return session.query(Sample).filter(Sample.id == sample_id).first()