import eventlet
eventlet.monkey_patch()  # Ensure async compatibility

import json
import time
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

BUFFER_FILE = "position.json"

@app.route('/')
def index():
    return "WebSocket Streaming Server Running!"

def watch_buffer_and_emit():
    """
    Background task that polls the JSON buffer for changes
    and emits the full hand object via Socket.IO.
    """
    last_left_hand = {"x": None, "y": None, "z": None}
    last_right_hand = {"x": None, "y": None, "z": None}
    last_left_gesture = "idle"
    last_right_gesture = "idle"
    last_left_timestamp = 0
    last_right_timestamp = 0

    while True:
        try:
            with open(BUFFER_FILE, "r") as f:
                data = json.load(f)
            # If data is None (e.g., file is empty), set defaults.
            if data is None:
                data = {
                    "left_hand": {
                        "position": last_left_hand,
                        "gesture": last_left_gesture,
                        "timestamp": last_left_timestamp
                    },
                    "right_hand": {
                        "position": last_right_hand,
                        "gesture": last_right_gesture,
                        "timestamp": last_right_timestamp
                    }
                }
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is missing or corrupted, use last known values.
            data = {
                "left_hand": {
                    "position": last_left_hand,
                    "gesture": last_left_gesture,
                    "timestamp": last_left_timestamp
                },
                "right_hand": {
                    "position": last_right_hand,
                    "gesture": last_right_gesture,
                    "timestamp": last_right_timestamp
                }
            }

        # Extract left hand data
        left_hand = data.get("left_hand", {})
        left_position = left_hand.get("position", last_left_hand)
        left_gesture = left_hand.get("gesture", last_left_gesture)
        left_timestamp = left_hand.get("timestamp", last_left_timestamp)

        # Extract right hand data
        right_hand = data.get("right_hand", {})
        right_position = right_hand.get("position", last_right_hand)
        right_gesture = right_hand.get("gesture", last_right_gesture)
        right_timestamp = right_hand.get("timestamp", last_right_timestamp)

        # Only emit if data has changed
        if (left_position != last_left_hand or left_gesture != last_left_gesture or left_timestamp != last_left_timestamp or
            right_position != last_right_hand or right_gesture != last_right_gesture or right_timestamp != last_right_timestamp):

            # Update last known values
            last_left_hand, last_right_hand = left_position, right_position
            last_left_gesture, last_right_gesture = left_gesture, right_gesture
            last_left_timestamp, last_right_timestamp = left_timestamp, right_timestamp

            # Emit the full hand object via Socket.IO
            socketio.emit("hand_update", {
                "left_hand": {
                    "position": left_position,
                    "gesture": left_gesture,
                    "timestamp": left_timestamp
                },
                "right_hand": {
                    "position": right_position,
                    "gesture": right_gesture,
                    "timestamp": right_timestamp
                }
            })

        time.sleep(0.01)  # Poll every 10ms

@socketio.on("connect")
def handle_connect():
    print("Client connected!")
    socketio.start_background_task(watch_buffer_and_emit)

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected!")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
