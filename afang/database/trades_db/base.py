import os
import pathlib

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

db_name = "trades.sqlite3"
base_dir = os.path.join(pathlib.Path(__file__).parents[3], "data", "trades")
engine = create_engine(f"sqlite:///{os.path.join(base_dir, db_name)}")

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

Base = declarative_base()
