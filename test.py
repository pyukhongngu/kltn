from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/sensor/data", methods=["POST"])
def receive_data():
    data = request.get_json()
    print("✅ Dữ liệu nhận được từ ESP32:")
    print(data)
    return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
