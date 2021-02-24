from sqlalchemy import Table, Column, MetaData
from sqlalchemy.types import Float, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy.dialects import postgresql

metadata = MetaData()

sample_table = Table("samples", metadata,
        Column('id', Integer, primary_key=True),
        Column("hash", String, unique=True),
        Column("fingerprint", postgresql.ARRAY(Float(precision=2), dimensions=2)),
    )

path_table = Table("sample_paths", metadata,
        Column('id', Integer, primary_key=True),
        Column('sample_id', Integer, ForeignKey("samples.id"), unique=True),
        Column('path', String)
    )

classes_table = Table("sample_classes", metadata,
        Column('id', Integer, primary_key=True),
        Column('sample_id', Integer, ForeignKey("samples.id"), unique=True),
        Column('sample_class', String)
    )

subclasses_table = Table("sample_subclasses", metadata,
        Column('id', Integer, primary_key=True),
        Column('sample_id', Integer, ForeignKey("samples.id"), unique=True),
        Column('sample_subclass', String)
    )

features_table = Table("features", metadata,
        Column('id', Integer, primary_key=True),
        Column('sample_id', Integer, ForeignKey("samples.id"), unique=True),
        Column("vgg", postgresql.ARRAY(Float(2), dimensions=1)),
        Column("yam", postgresql.ARRAY(Float(2), dimensions=1)),
        Column("mfcc", postgresql.ARRAY(Float(2), dimensions=2)),
        Column("hardness", Float(2)),
        Column("depth", Float(2)),
        Column("brightness", Float(2)),
        Column("roughness", Float(2)),
        Column("warmth", Float(2)),
        Column("sharpness", Float(2)),
        Column("boominess", Float(2)),
        Column("contrast", postgresql.ARRAY(Float(2), dimensions=2)),
        Column("zero_crossing_rate", postgresql.ARRAY(Float(2), dimensions=1)),
        Column("spectral_flatness", postgresql.ARRAY(Float(2), dimensions=1))
    )

def create_tables(engine, drop=False):
    if drop:
        classes_table.drop(engine, checkfirst=True)
        subclasses_table.drop(engine, checkfirst=True)
        path_table.drop(engine, checkfirst=True)
        features_table.drop(engine, checkfirst=True)
        sample_table.drop(engine, checkfirst=True)

    metadata.create_all(engine)