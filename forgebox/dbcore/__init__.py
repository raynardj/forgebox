from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from ..config import SQLALCHEMY_DATABASE_URI

Base = automap_base()

import os


def create_db():
    print("Initialize forge db")
    from forge.app import db
    db.create_all()
    os.environ["FORGE_DB_INITIALIZED"] = "1"

if ("FORGE_DB_INITIALIZED" in os.environ) ==False: #no key
    create_db()
else: # has key
    if os.environ["FORGE_DB_INITIALIZED"] != '1':
        create_db()


engine = create_engine(SQLALCHEMY_DATABASE_URI)
Base.prepare(engine, reflect=True)

session = Session(engine)

from .ml_base import taskModel, dataFormat, hyperParam, hyperParamLog, weightModel, metricModel, metricLog, trainModel, \
    logModel, keyMetricModel