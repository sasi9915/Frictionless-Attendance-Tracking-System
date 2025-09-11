from flask import Flask, jsonify, render_template
from flask_cors import CORS
from threading import Lock
from counters_shared import zone_counters, counter_lock

# Lazy import/start to avoid double running on module import
from threading import Thread
detection_booted = False

def ensure_detection():
    global detection_booted
    if detection_booted:
        return
    from Multiple_zones import start_detection  # import inside to avoid early heavy init
    start_detection()
    detection_booted = True

app = Flask(__name__, template_folder='templates')  # Specify the templates folder
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/')
def index():
    ensure_detection()
    return render_template('index.html')  # Render the index.html file

@app.route('/api/zone_counters')
def get_zone_counters():
    with counter_lock:
        return jsonify(zone_counters)

if __name__ == '__main__':
    # IMPORTANT: disable reloader to prevent double start of detection threads
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)