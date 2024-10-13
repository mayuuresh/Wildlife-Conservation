import os
from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/start-detection')
def start_detection():
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.Popen(['python', 'yolo_detection.py'])
        return jsonify({"message": "Detection started. Check the webcam feed."})
    except Exception as e:
        return jsonify({"message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
