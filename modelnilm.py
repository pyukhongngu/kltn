from Database import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, unique=True)
    Name = Column(String)
    Email = Column(String)
    Password = Column(String)

class Data(Base):
    __tablename__ = "datas"

    index = Column(Integer, primary_key=True, unique=True, autoincrement=True)
    Power = Column(Float)
    Irms = Column(Float)
    Label = Column(Integer)
    Time = Column(DateTime, default = datetime.now)
    users_id = Column(Integer, default=1)
    
class UsersDevice(Base):
    __tablename__ = "userdevices"

    id = Column(Integer, primary_key=True, unique= True)
    Name = Column(String)
    Company = Column(String)
    Label = Column(Integer)





