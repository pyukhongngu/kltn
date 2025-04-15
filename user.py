from fastapi import APIRouter, Depends, HTTPException, status
from Database import SessionLocal
from modelnilm import User
from typing import Annotated
from sqlalchemy.orm import Session
import smtplib
import random
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

OTP = {}
RegisterEmail = {}

# Hàm gửi email và mã otp
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


# database access dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Router
router = APIRouter(prefix="/user")

# database dependency
db_dependency = Annotated[Session, Depends(get_db)]

# API cho việc đăng ký
@router.post("/register")
def register_user(Name: str, Email: str, Password: str, db: db_dependency):
    db_user = db.query(User).filter(User.Email == Email).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    error = send_email(Email)
    RegisterEmail[Email] = {"Name": Name, "Password": Password}
    if error != True:
        if "550" in error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email does not exist")
        print(str(error))
    return HTTPException(status_code=status.HTTP_200_OK, detail="OTP code sent to your email")

# API cho việc đăng nhập
@router.post("/login")
def login_user(Email: str, Password: str, db: db_dependency):
    db_user = db.query(User).filter(User.Email == Email).first()
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if Password != db_user.Password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect password")
    return HTTPException(status_code=status.HTTP_200_OK, detail="Login successful")

# API cho việc xác nhập OTP và tạo tài khoản
@router.post("/verify-otp")
def verify_otp(Email: str, otp: str, db: db_dependency):
    if Email in OTP and OTP[Email] == otp:    
        new_user = User(Name=RegisterEmail[Email]["Name"], Email=Email, Password=RegisterEmail[Email]["Password"])
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        del RegisterEmail[Email]
        del OTP[Email]
        return HTTPException(status_code=status.HTTP_200_OK, detail="Register successful")
    else:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect OTP code")

# API cho việc cập nhật mật khẩu
@router.post("/update_password")
def update_password(Email: str, db: db_dependency):
    db_user = db.query(User).filter(User.Email == Email).first()
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User have not registered")
    send_email(Email)
    return HTTPException(status_code=status.HTTP_200_OK, detail="OTP code sent to your email")

# API cho việc xác nhận OTP và cập nhật mật khẩu
@router.post("/verify_otp_update")
def verify_otp_update_password(Email: str, otp: str, db: db_dependency):
    if Email in OTP and OTP[Email] == otp:
        del OTP[Email]
        return HTTPException(status_code=status.HTTP_200_OK, detail="Correct OTP")
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect OTP code")

@router.post("/update_password_final")
def update_password(Email: str, Password: str, db: db_dependency):
    db_user = db.query(User).filter(User.Email == Email).first()
    db_user.Password = Password
    db.commit()
    return HTTPException(status_code=status.HTTP_200_OK, detail="Password updated successfully")





    
