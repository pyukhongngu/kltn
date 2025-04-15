from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import joblib
import numpy as np
from keras.models import load_model
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Annotated

# =================== Init FastAPI App ===================
app = FastAPI()

# =================== Load Model ===================
def load_lstm_model(model_path="lstm_power_model.h5", scaler_path="scaler.gz"):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_lstm_model()
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

# =================== MongoDB Setup ===================
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

DATABASE_URL = "mongodb+srv://admin:admin@app.4nbgnsg.mongodb.net/?retryWrites=true&w=majority&appName=app"
client = AsyncIOMotorClient(DATABASE_URL)
db = client.nilm_users  

# =================== OTP Email ===================
OTP = {}
RegisterEmail = {}

def send_email(receiver_email: str):
    sender_email = "mylabnilm@gmail.com"
    sender_password = "xrfm ramf nbwb aqol"
    otp = str(random.randint(100000, 999999))

    subject = "Your OTP Code"
    body = f"Your OTP code for the app is: {otp}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    OTP[receiver_email] = otp

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        return str(e)

# =================== Dependency ===================
# MongoDB does not need to close like SQLAlchemy
async def get_db():
    return db

db_dependency = Annotated[AsyncIOMotorClient, Depends(get_db)]

# =================== Pydantic Models ===================
class RegisterRequest(BaseModel):
    Name: str
    Email: str
    Password: str

class OTPRequest(BaseModel):
    Email: str
    otp: str

class LoginRequest(BaseModel):
    Email: str
    Password: str

class SensorData(BaseModel):
    date: str
    time: str
    global_active_power: float
    global_reactive_power: float
    voltage: float
    global_intensity: float
    sub_metering_1: float
    sub_metering_2: float
    sub_metering_3: float

class SensorPredictionResponse(BaseModel):
    prediction: float

# =================== User Router ===================
router = APIRouter(prefix="/user")

@router.post("/register")
async def register_user(request: RegisterRequest, db: db_dependency):
    db_user = await db.users.find_one({"Email": request.Email})
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    error = send_email(request.Email)
    RegisterEmail[request.Email] = {
        "Name": request.Name,
        "Password": request.Password
    }
    if error != True:
        if "550" in error:
            raise HTTPException(status_code=400, detail="Email does not exist")
    return {"message": "OTP code sent to your email"}

@router.post("/verify-otp")
async def verify_otp(request: OTPRequest, db: db_dependency):
    if request.Email in OTP and OTP[request.Email] == request.otp:
        new_user = {
            "Name": RegisterEmail[request.Email]["Name"],
            "Email": request.Email,
            "Password": RegisterEmail[request.Email]["Password"]
        }
        await db.users.insert_one(new_user)
        del RegisterEmail[request.Email]
        del OTP[request.Email]
        return {"message": "Register successful"}
    raise HTTPException(status_code=400, detail="Incorrect OTP code")

@router.post("/login")
async def login_user(request: LoginRequest, db: db_dependency):
    db_user = await db.users.find_one({"Email": request.Email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user['Password'] != request.Password:
        raise HTTPException(status_code=400, detail="Incorrect password")
    return {"message": "Login successful"}

@router.post("/update_password")
async def update_password(request: OTPRequest, db: db_dependency):
    db_user = await db.users.find_one({"Email": request.Email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not registered")
    send_email(request.Email)
    return {"message": "OTP code sent to your email"}

@router.post("/verify_otp_update")
async def verify_otp_update_password(request: OTPRequest, db: db_dependency):
    if request.Email in OTP and OTP[request.Email] == request.otp:
        del OTP[request.Email]
        return {"message": "Correct OTP"}
    raise HTTPException(status_code=400, detail="Incorrect OTP code")

@router.post("/update_password_final")
async def update_password_final(request: OTPRequest, Password: str, db: db_dependency):
    db_user = await db.users.find_one({"Email": request.Email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.users.update_one({"Email": request.Email}, {"$set": {"Password": Password}})
    return {"message": "Password updated successfully"}

# =================== Middleware ===================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# =================== Sensor Endpoints ===================
@router.post("/sensor/predict", response_model=SensorPredictionResponse)
async def predict_sensor_data(data: SensorData = None):
    try:
        value = data.global_active_power if data else np.random.uniform(0.1, 6.0)

        input_data = np.array([[value]], dtype=np.float32)
        scaled_data = scaler.transform(input_data)
        lstm_input = scaled_data.reshape((1, 1, 1))
        prediction = model.predict(lstm_input)
        prediction_real = scaler.inverse_transform(prediction)

        return {"prediction": float(prediction_real[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# =================== Run App ===================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
