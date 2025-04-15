from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import Annotated
import pandas as pd
from datetime import datetime
import paho.mqtt.client as mqtt

from Database import engine, SessionLocal
from modelnilm import Data, UsersDevice

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# Data models
def create_data_model(power: float, label: int, irms: float, time: datetime, users_id: int) -> Data:
    return Data(
        Power=power,
        Label=label,
        Irms=irms,
        Time=time,
        users_id = users_id
    )

# Database operations
def add_data(df: pd.DataFrame, db: Session):
    for i in range(1, len(df)):
        data_model = create_data_model(
            power=df.iloc[i, 1],
            label=df.iloc[i, 0],
            irms=df.iloc[i, 2],
            time=datetime.strptime(df.iloc[i, 3], "%Y-%m-%d %H:%M:%S.%f"),
            users_id=1
        )
        db.add(data_model)
    db.commit()

def add_device(db: Session):

    devices = [
        {"Name": "SacDienThoai", "Company": "Iphone", "Label": 0},
        {"Name": "SacMayTinh", "Company": "Hp", "Label": 1},
        {"Name": "MaySay", "Company": "", "Label": 2},
        {"Name": "MayEp", "Company": "Savtm", "Label": 3},
        {"Name": "MayLockk", "Company": "BlueStone", "Label": 4},
        {"Name": "TuLanh", "Company": "Funiki", "Label": 5},
        {"Name": "Quat", "Company": "QuatCo", "Label": 6},
        {"Name": "Led", "Company": "Vianco", "Label": 7}
    ]
    
    for device in devices:
        device_model = UsersDevice(
            Name=device["Name"],
            Company=device["Company"],
            Label=device["Label"]
        )
        db.add(device_model)
    db.commit()

def main():
    # Read data
    df = pd.read_csv("power_data_week_v2.csv", header=None)
    
    # Initialize database
    with next(get_db()) as db:
        add_device(db)
        #add_data(df, db)

if __name__ == "__main__":
    main()
