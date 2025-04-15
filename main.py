from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Annotated
import paho.mqtt.client as mqtt
from datetime import datetime

import modelnilm
import data as data
import user as user
from Database import engine, SessionLocal
from modelnilm import Data, UsersDevice

# MQTT Configuration
MQTT_CONFIG = {
    "broker": "gerbil-01.rmq.cloudamqp.com",  # Địa chỉ mới
    "port": 1883,
    "username": "zakprdkd:zakprdkd",
    "password": "vvQGkzs0GwxNwgyDPEQKyb0v7l4YHwLq",
    "topic": "data/#"  # Có thể thay đổi topic nếu cần
}
# Global variables
power = 0
label = 0
irms = 0
previous_label = 0
label_change_history = []
# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# MQTT callbacks
def on_message(client, userdata, msg):
    global power, label, irms, previous_label, label_change_history
    print(f"{msg.topic},{msg.payload.decode()}")
    
    payload = msg.payload.decode()
    print(payload)
    power_data = payload.split(",")
    label = int(power_data[0])
    power = float(power_data[1])
    irms = float(power_data[2])
    if previous_label != label:
        label_change_history.append((label, previous_label, datetime.now()))
        previous_label = label
    # Store in database
    db: Session = next(get_db())
    data_model = Data(
        Power=power,
        Label=label,
        Irms=irms
    )
    db.add(data_model)
    db.commit()
    db.close()

# Initialize MQTT client
def setup_mqtt_client():
    client = mqtt.Client()
    client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
    client.user_data_set(None)
    client.on_message = on_message
    client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
    client.loop_start()
    client.subscribe(MQTT_CONFIG["topic"])
    return client

# Initialize FastAPI
def create_app():
    app = FastAPI()
    
    # CORS middleware configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(data.router)
    return app

# Main initialization
app = create_app()
mqtt_client = setup_mqtt_client()
modelnilm.Base.metadata.create_all(bind=engine)

@app.get("/device_on_off")
def device_on_off(db: db_dependency):
    global label_change_history
    if label_change_history:
        response = {}
        num_devices = 8
        device_model = db.query(UsersDevice).all()
        device_info = [""] * num_devices
        device_list_change = []
        for device_entry in device_model:
            device_info[device_entry.Label] = f"{device_entry.Name}".strip()

        for record in label_change_history:
            label_1, previous_label_1, time_1 = record[0], record[1], record[2]
            time_1 = time_1.strftime("%Y-%m-%d %H:%M")
            label_1 = format(label_1, f'0{num_devices}b')
            label_1 = [int(bit) for bit in label_1]
            previous_label_1 = format(previous_label_1, f'0{num_devices}b')
            previous_label_1 = [int(bit) for bit in previous_label_1]
            for i in range(len(label_1)):
                if label_1[i] != previous_label_1[i]:
                    if label_1[i] == 1:
                        device_list_change.append((device_info[i], "On", time_1))
                    else:
                        device_list_change.append((device_info[i], "Off", time_1))

        for j in range(len(device_list_change)):
            response[j] = {"Device": device_list_change[j][0], "Status": device_list_change[j][1], "Time": device_list_change[j][2]}
        label_change_history = []
    else:
        response = {"message": "No device change"}
    return response

