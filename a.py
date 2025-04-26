from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    print("Received:", data)
    return jsonify({"status": "ok", "received": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
