from fastapi import APIRouter, Depends, HTTPException, status
from Database import SessionLocal
from modelnilm import Data, User, UsersDevice
from datetime import timedelta, datetime
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import desc

# API Router
router = APIRouter(prefix="/data")

# db dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
db_dependency = Annotated[Session, Depends(get_db)]

# Hàm round_by_day và round_today
def round_by_day(dt: datetime):
    hour = dt.hour
    minute = dt.minute
    if minute < 15:
        minute = 0
    elif minute < 45:
        minute = 30
    else:
        minute = 0
        hour += 1 
    if hour == 24:
        hour = 0
        dt += timedelta(days = 1)
    dt = dt.replace(hour = hour, minute = minute, second = 0, microsecond = 0)
    return dt.strftime("%H:%M")

def round_today(dt: datetime):
    return dt.replace(hour = 0, minute = 0, second = 0, microsecond = 0)


# Thông số hiện tại (Công suất, Irms, Điện năng tiêu thụ), Thiết bị nào bật tắt
@router.get("/current_data")
async def get_current_data(db: db_dependency, id: int=1):
    data_model = db.query(Data).filter(Data.users_id == id).order_by(desc(Data.Time)).first()
    if not data_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data available yet")

    # Get device information
    num_devices = 8
    device_model = db.query(UsersDevice).all()
    device_info = ["Unknown Device"] * num_devices
    for device in device_model:
        device_info[device.Label] = f"{device.Name}".strip()

    # Process device states
    label = format(data_model.Label, f'0{num_devices}b')
    states = ["ON" if bit == "1" else "OFF" for bit in label]

    # Build response
    response = {
        device_info[i]: states[i] for i in range(num_devices) if device_info[i] != "Unknown Device"
    }
    response["Power"] = f'{format(data_model.Power, ".2f")}W'
    response["Energy"] = f'{format((data_model.Power) * (10 / 3600), ".2f")}Wh'

    return response

# Tính toán điện năng tiêu thụ theo ngày
@router.get("/history/days")
async def get_data_by_day(db: db_dependency, id: int=1):
    now = datetime.now()
    day_start = now - timedelta(hours=24)
    day_start = day_start.replace(minute=0, second=0, microsecond=0)
    
    total_energy = {}
    hours_per_interval = 3
    num_intervals = 8

    for i in range(num_intervals):
        start_time = day_start + timedelta(hours=hours_per_interval * i)
        end_time = start_time + timedelta(hours=hours_per_interval)

        # Query data for this time interval
        data_records = db.query(Data).filter(
            Data.Time >= start_time,
            Data.Time < end_time,
            Data.users_id == id
        ).all()

        # Calculate average power for this interval
        energy = sum((record.Power / 1000) * (10 / 3600) for record in data_records)
        interval_midpoint = start_time + timedelta(hours=hours_per_interval/2)
        time_key = round_by_day(interval_midpoint)

        total_energy[time_key] = format(
            energy, ".2f"
        ) if energy else "0.00"

    return total_energy

# Tính điện năng tiêu thụ theo tuần
@router.get("/history/weeks")
async def get_data_by_week(db: db_dependency, id: int=1):
    
    today = round_today(datetime.now())
    total_energy = {}
    
    # Last 7 days data (daily averages)
    week_start = today - timedelta(days=7)
    for i in range(7):
        start_time = week_start + timedelta(days=i)
        end_time = start_time + timedelta(days=1)
        
        data_records = db.query(Data).filter(
            Data.Time >= start_time,
            Data.Time < end_time,
            Data.users_id == id
        ).all()

        energy = sum((record.Power / 1000) * (10 / 3600) for record in data_records)
        date_key = start_time.strftime("%d-%m")
        
        total_energy[date_key] = format(
            energy, ".2f"
        ) if energy else "0.00"

    return total_energy


# Thông tin thiết bị
@router.get("/info/device")
def device_info(db: db_dependency):
    device_model = db.query(UsersDevice).all()
    response = {}
    for device in device_model:
        response[device.Name] = device.Company
    return response


# Thời gian thiết bị bật (theo giờ)
@router.get("/info/device_time")
def device_on_time(db: db_dependency, id: int=1):
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    # Get data records for the specified time period
    data_records = db.query(Data).filter(
        Data.Time > start_time,
        Data.Time < end_time,
        Data.users_id == id
    ).all()
    
    if not data_records:
        return {"message": "No data available for the last 24 hours"}

    # Get device information
    num_devices = 8
    device_model = db.query(UsersDevice).all()
    device_info = ["Unknown Device"] * num_devices
    for device_entry in device_model:
        device_info[device_entry.Label] = f"{device_entry.Name}".strip()

    # Calculate usage time for each device
    label_values = [entry.Label for entry in data_records]
    time_devices = [0] * num_devices
    
    for label in label_values:
        label = format(label, f'0{num_devices}b')
        label = [int(bit) for bit in label]
        for i in range(len(label)):
            if label[i] == 1:
                time_devices[i] += 10

    # Convert to hours with one decimal place
    time_devices = [round(time/60, 1) for time in time_devices]

    # Build response with only known devices
    response = {
        device_info[i]: time_devices[i]
        for i in range(num_devices) if device_info[i] != "Unknown Device"
    }
    return response

# Thêm bớt thiết bị vào database
@router.post("/add_device")
def add_device(Name: str, Company: str, db: db_dependency):
    device_model = db.query(UsersDevice).order_by(desc(UsersDevice.Label)).first()
    new_label = device_model.Label + 1
    new_device = UsersDevice(Name=Name, Company=Company, Label=new_label)
    db.add(new_device)
    db.commit()
    db.refresh(new_device)
    return {"message": "Device added successfully"}

