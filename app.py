from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import smtplib
import random
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib
import numpy as np
from keras.models import load_model
import csv
from pathlib import Path
from datetime import datetime
import logging


# =================== Init Flask App ===================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///./kltn.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
logging.basicConfig(level=logging.DEBUG)

CORS(app, resources={r"/*": {"origins": "*"}})
# =================== Load Model ===================
def load_lstm_model(model_path="lstm_power_model.h5", scaler_path="scaler.gz"):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_lstm_model()
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

# =================== Database Setup ===================
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, index=True)
    Name = db.Column(db.String, nullable=False)
    Email = db.Column(db.String, unique=True, index=True, nullable=False)
    Password = db.Column(db.String, nullable=False)

class SensorDataModel(db.Model):
    __tablename__ = "sensor_data"
    id = db.Column(db.Integer, primary_key=True, index=True)
    date = db.Column(db.String)
    time = db.Column(db.String)
    global_active_power = db.Column(db.Float)
    global_reactive_power = db.Column(db.Float)
    voltage = db.Column(db.Float)
    global_intensity = db.Column(db.Float)
    sub_metering_1 = db.Column(db.Float)
    sub_metering_2 = db.Column(db.Float)
    sub_metering_3 = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class SensorDataHistory(db.Model):
    __tablename__ = "sensor_data_history"
    id = db.Column(db.Integer, primary_key=True, index=True)
    date = db.Column(db.String)
    time = db.Column(db.String)
    global_active_power = db.Column(db.Float)
    global_reactive_power = db.Column(db.Float)
    voltage = db.Column(db.Float)
    global_intensity = db.Column(db.Float)
    sub_metering_1 = db.Column(db.Float)
    sub_metering_2 = db.Column(db.Float)
    sub_metering_3 = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    action_type = db.Column(db.String)  
    original_record_id = db.Column(db.Integer) 


# Ensure database tables are created within the app context
with app.app_context():
    db.create_all()

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

# =================== User Endpoints ===================
@app.route('/user/register', methods=['POST'])
def register_user():
    data = request.get_json()
    name = data.get('Name')
    email = data.get('Email')
    password = data.get('Password')

    # Check if the user already exists
    db_user = User.query.filter_by(Email=email).first()
    if db_user:
        abort(400, description="Email already registered")

    # Send OTP to email
    error = send_email(email)
    RegisterEmail[email] = {
        "Name": name,
        "Password": password
    }

    if error != True:
        if "550" in error:
            abort(400, description="Email does not exist")
    
    return jsonify({"message": "OTP code sent to your email"})

@app.route('/user/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('Email')
    otp = data.get('otp')

    if email in OTP and OTP[email] == otp:
        new_user = User(
            Name=RegisterEmail[email]["Name"],
            Email=email,
            Password=RegisterEmail[email]["Password"]
        )
        db.session.add(new_user)
        db.session.commit()
        del RegisterEmail[email]
        del OTP[email]
        return jsonify({"message": "Register successful"})
    
    abort(400, description="Incorrect OTP code")

@app.route('/user/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email = data.get('Email')
    password = data.get('Password')

    db_user = User.query.filter_by(Email=email).first()
    if not db_user:
        abort(404, description="User not found")
    if db_user.Password != password:
        abort(400, description="Incorrect password")
    
    return jsonify({"message": "Login successful"})

@app.route('/user/update_password', methods=['POST'])
def update_password():
    data = request.get_json()
    email = data.get('Email')
    error = send_email(email)
    return jsonify({"message": "OTP code sent to your email"})

@app.route('/user/verify_otp_update', methods=['POST'])
def verify_otp_update_password():
    data = request.get_json()
    email = data.get('Email')
    otp = data.get('otp')
    
    if email in OTP and OTP[email] == otp:
        del OTP[email]
        return jsonify({"message": "Correct OTP"})
    
    abort(400, description="Incorrect OTP code")

@app.route('/user/update_password_final', methods=['POST'])
def update_password_final():
    data = request.get_json()
    email = data.get('Email')
    password = data.get('Password')
    
    db_user = User.query.filter_by(Email=email).first()
    if not db_user:
        abort(404, description="User not found")
    
    db_user.Password = password
    db.session.commit()
    
    return jsonify({"message": "Password updated successfully"})

# =================== Sensor Endpoints ===================
@app.route('/sensor/predict', methods=['POST'])
def predict_sensor_data():
    data = request.get_json()
    print("Received data:", data)  # In ra dữ liệu nhận được từ ESP32
    
    try:
        # Lấy giá trị 'global_active_power', nếu không có thì lấy giá trị ngẫu nhiên
        value = data.get('global_active_power', np.random.uniform(0.1, 6.0))
        input_data = np.array([[value]], dtype=np.float32)

        # Chuyển đổi dữ liệu đầu vào
        scaled_data = scaler.transform(input_data)
        lstm_input = scaled_data.reshape((1, 1, 1))

        # Dự đoán
        prediction = model.predict(lstm_input)
        prediction_real = scaler.inverse_transform(prediction)

        # Lấy thời gian dự đoán
        prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Nhận dữ liệu từ ESP32 và lưu vào cơ sở dữ liệu
        sensor_data_history = SensorDataHistory(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            global_active_power=value,
            global_reactive_power=data.get('global_reactive_power', 0),
            voltage=data.get('voltage', 0),
            global_intensity=data.get('global_intensity', 0),
            sub_metering_1=data.get('sub_metering_1', 0),
            sub_metering_2=data.get('sub_metering_2', 0),
            sub_metering_3=data.get('sub_metering_3', 0),
            action_type="predict",
            original_record_id=None
        )
        db.session.add(sensor_data_history)
        db.session.commit()

        # Trả về kết quả dự đoán và thời gian dự đoán
        return jsonify({
            "prediction": float(prediction_real[0][0]),
            "prediction_time": prediction_time
        })

    except Exception as e:
        print("Error:", str(e))  # In ra lỗi nếu có
        abort(500, description=f"Prediction failed: {str(e)}")

@app.route('/sensor/history', methods=['GET'])
def get_sensor_history():
    try:
        # Lấy tất cả dữ liệu từ SensorDataHistory
        sensor_history = SensorDataHistory.query.all()

        # Chuyển đổi thành JSON
        result = [{
            "id": item.id,
            "date": item.date,
            "time": item.time,
            "global_active_power": item.global_active_power,
            "global_reactive_power": item.global_reactive_power,
            "voltage": item.voltage,
            "global_intensity": item.global_intensity,
            "sub_metering_1": item.sub_metering_1,
            "sub_metering_2": item.sub_metering_2,
            "sub_metering_3": item.sub_metering_3,
            "timestamp": item.timestamp,
            "action_type": item.action_type,
            "original_record_id": item.original_record_id
        } for item in sensor_history]

        return jsonify(result)
    except Exception as e:
        abort(500, description=f"Error retrieving history: {str(e)}")


@app.route('/sensor/predict-range', methods=['GET', 'POST'])
def predict_range():
    if request.method == 'POST':
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
    else:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

    logging.debug(f"Start Date: {start_date}, End Date: {end_date}")

    file_path = Path("text.txt")
    if not file_path.exists():
        logging.error(f"File {file_path} không tồn tại")
        abort(404, description="File text.txt không tồn tại")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        if not rows:
            logging.warning("File không có dữ liệu")
            abort(400, description="File không có dữ liệu")

        logging.debug(f"Số lượng dữ liệu trong file: {len(rows)}")
        
        filtered_rows = [row for row in rows if start_date <= row["Date"] <= end_date]
        logging.debug(f"Số lượng dữ liệu sau khi lọc: {len(filtered_rows)}")

        if not filtered_rows:
            logging.warning(f"Không có dữ liệu trong khoảng thời gian từ {start_date} đến {end_date}")
            abort(400, description="Không có dữ liệu trong khoảng thời gian này")

        predictions = []
        for row in filtered_rows:
            try:
                value = float(row["Global_active_power"])
                input_data = np.array([[value]], dtype=np.float32)
                scaled_data = scaler.transform(input_data)
                lstm_input = scaled_data.reshape((1, 1, 1))
                prediction = model.predict(lstm_input)
                prediction_real = scaler.inverse_transform(prediction)
                
                prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                predictions.append({
                    "Date": row["Date"],
                    "Time": row["Time"],
                    "Predicted": float(prediction_real[0][0]),
                    "Prediction Time": prediction_time
                })

            except Exception as e:
                logging.error(f"Error predicting for row {row}: {str(e)}")
        
        return jsonify({"predictions": predictions})

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        abort(500, description=f"Lỗi xử lý file: {str(e)}")

csv.field_size_limit(2**20)

@app.route('/sensor/predict-from-file', methods=['GET'])
def predict_from_file():
    file_path = Path("text.txt")
    if not file_path.exists():
        abort(404, description="File text.txt không tồn tại")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        if not rows:
            abort(400, description="File không có dữ liệu")

        latest = rows[-1]
        value = float(latest["Global_active_power"])
        input_data = np.array([[value]], dtype=np.float32)
        scaled_data = scaler.transform(input_data)
        lstm_input = scaled_data.reshape((1, 1, 1))
        prediction = model.predict(lstm_input)
        prediction_real = scaler.inverse_transform(prediction)

        return jsonify({"prediction": float(prediction_real[0][0])})
    except Exception as e:
        abort(500, description=f"Lỗi xử lý file: {str(e)}")


@app.route('/sensor/get-inputs-from-file', methods=['GET'])
def get_inputs_from_file():
    file_path = Path("text.txt")
    if not file_path.exists():
        abort(404, description="File text.txt không tồn tại")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        if not rows:
            abort(400, description="File không có dữ liệu")

        inputs = [float(row["Global_active_power"]) for row in rows]

        return jsonify({"inputs": inputs})

    except Exception as e:
        abort(500, description=f"Lỗi xử lý file: {str(e)}")

@app.route('/sensor/get-data-from-file', methods=['GET'])
def get_data_from_file():
    file_path = Path("text.txt")
    if not file_path.exists():
        abort(404, description="File text.txt không tồn tại")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        if not rows:
            abort(400, description="File không có dữ liệu")
        data = [{
            "Date": row["Date"],
            "Time": row["Time"],
            "Global_active_power": row["Global_active_power"],
            "Global_reactive_power": row["Global_reactive_power"],
            "Voltage": row["Voltage"],
            "Global_intensity": row["Global_intensity"],
            "Sub_metering_1": row["Sub_metering_1"],
            "Sub_metering_2": row["Sub_metering_2"],
            "Sub_metering_3": row["Sub_metering_3"]
        } for row in rows]

        return jsonify({"data": data})

    except Exception as e:
        abort(500, description=f"Lỗi xử lý file: {str(e)}")
def generate_fake_data():
    # Giả sử dữ liệu điện năng dao động từ 0.3 kWh đến 0.6 kWh trong 30 ngày
    return [round(random.uniform(0.3, 0.6), 3) for _ in range(30)]

# API endpoint để dự đoán tiêu thụ điện năng
@app.route('/predict', methods=['GET'])
def predict():
    # Tạo dữ liệu giả ngẫu nhiên cho 30 ngày gần nhất
    last_30_days = np.array(generate_fake_data())

    # Tải model và scaler
    model, scaler = load_lstm_model()

    # Copy chuỗi để sử dụng
    input_seq = last_30_days.copy()
    predictions = []

    # Dự đoán 30 ngày tiếp theo
    for _ in range(30):
        reshaped = np.reshape(input_seq, (1, 30, 1))
        pred = model.predict(reshaped, verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred)

    # Chuyển sang dạng chuẩn để hiển thị
    predictions = np.array(predictions).reshape(-1, 1)
    real_predictions = scaler.inverse_transform(predictions)

    # Tạo kết quả dưới dạng JSON
    result = [{"day": i+1, "predicted_kWh": float(val[0])} for i, val in enumerate(real_predictions)]

    # Trả kết quả dự đoán
    return jsonify(result)



@app.route('/sensor/data', methods=['POST'])
def receive_sensor_data():
    data = request.get_json()
    try:
        date = data['Date']
        time = data['Time']
        global_active_power = data['Global_active_power']
        global_reactive_power = data['Global_reactive_power']
        voltage = data['Voltage']
        global_intensity = data['Global_intensity']
        sub_metering_1 = data['Sub_metering_1']
        sub_metering_2 = data['Sub_metering_2']
        sub_metering_3 = data['Sub_metering_3']

        new_data = SensorDataModel(
            date=date, time=time, global_active_power=global_active_power,
            global_reactive_power=global_reactive_power, voltage=voltage,
            global_intensity=global_intensity, sub_metering_1=sub_metering_1,
            sub_metering_2=sub_metering_2, sub_metering_3=sub_metering_3
        )
        db.session.add(new_data)
        db.session.commit()

        return jsonify({"message": "Data received and saved successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


# =================== Run App ===================
if __name__ == '__main__':
    def before_first_request():
        print("Initializing app...") 
    
    app.run(host='0.0.0.0', port=8000, debug=True)

