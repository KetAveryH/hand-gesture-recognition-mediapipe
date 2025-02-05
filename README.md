# hand-gesture-recognition-using-mediapipe (Modified for Custom Application)

This project is a modified version of the original [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).  
The code has been repurposed to create a new application with API endpoints, omitting the original machine learning models. These may be included at a later date if deemed necessary.


This repository now includes:
- **Demo application:**  
  `app.py` has been refactored to use a separate canvas module (`canvas.py`) for all OpenCV drawing and display functions.
- **Streaming server:**  
  `server.py` continuously reads hand data from a JSON buffer file (`position.json`) and emits updates via Socket.IO.
- **API Integration:**  
  Added RESTful API endpoints for real-time hand gesture data streaming.
- **Modular Architecture:**  
  Refactored codebase for flexibility and performance.

---

## Requirements

- Python 3.6 or later
- [mediapipe](https://pypi.org/project/mediapipe/) (0.8.1 or later)
- [OpenCV](https://pypi.org/project/opencv-python/) (3.4.2 or later)
- [Flask](https://pypi.org/project/Flask/)
- [Flask-SocketIO](https://pypi.org/project/Flask-SocketIO/)
- [Eventlet](https://pypi.org/project/eventlet/)
- Additional packages as specified in [`requirements.txt`](requirements.txt)

---

## Installation

It is recommended to use a virtual environment. For example, using `venv`:

1. **Create and activate a virtual environment:**

   **On Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install all required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Demo

### Running the Hand Gesture Recognition Demo

The demo application has been refactored into modular code. The display logic is now separated into `canvas.py` (providing a `Canvas` class with a `display()` method), and is used by `app.py`. To run the demo using your webcam, execute:

```bash
python app.py
```

The following options can be specified when running the demo:
- **`--device`**  
  Specify the camera device number (Default: 0)
- **`--width`**  
  Capture width (Default: 960)
- **`--height`**  
  Capture height (Default: 540)
- **`--use_static_image_mode`**  
  Whether to use the static image mode for MediaPipe inference (Default: False)
- **`--min_detection_confidence`**  
  Detection confidence threshold (Default: 0.8)
- **`--min_tracking_confidence`**  
  Tracking confidence threshold (Default: 0.5)

### Running the Streaming Server

The streaming server reads hand data from the JSON buffer file (`position.json`) and emits updates via Socket.IO. To run the server, execute:

```bash
python server.py
```

When running, the server will print connection events and emit hand data updates whenever changes occur.

---

### Viewing API Outputs

The `socket.html` file loads hand data from the JSON buffer file (`position.json`) and displays their API data along with a red and blue ball on the web-page, intended to represent the tracking of your hands.

## Directory Structure

```
hand-gesture-recognition-using-mediapipe/
│  app.py
│  canvas.py
│  server.py
│  requirements.txt
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │      keypoint.csv
│  │      keypoint_classifier.hdf5
│  │      keypoint_classifier.py
│  │      keypoint_classifier.tflite
│  │      keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│         point_history.csv
│         point_history_classifier.hdf5
│         point_history_classifier.py
│         point_history_classifier.tflite
│         point_history_classifier_label.csv
│          
└─utils
       cvfpscalc.py
```

- **app.py:**  
  The main application for hand gesture inference. It uses `canvas.py` for all drawing and display tasks.
  
- **canvas.py:**  
  Contains the `Canvas` class that encapsulates all OpenCV display/drawing functionality (e.g., a `canvas.display(event)` method).

- **server.py:**  
  A Socket.IO streaming server that continuously reads hand state data from `position.json` and emits it to connected clients.

- **API Endpoints:**  
  Provides real-time hand gesture data through RESTful APIs.

- **utils/cvfpscalc.py:**  
  Module for calculating FPS.

---

## About This Fork

This fork focuses on:
- Removing the original ML models
- Refactoring the code for modular architecture
- Adding RESTful API endpoints for hand gesture data streaming
- Enhancing performance for real-time applications

The original project remains an excellent resource for gesture recognition using MediaPipe and machine learning models.  
This version is tailored for real-time API-driven gesture control systems.

---

## Original Author

- Kazuhito Takahashi ([Twitter](https://twitter.com/KzhtTkhs))

## Contributors

- Nikita Kiselov ([GitHub](https://github.com/kinivi)) – Translation and Documentation
- **Ket Hollingsworth** – Application Development, API Integration, Code Refactoring

---

## License

This project is licensed under the [Apache v2 license](LICENSE).

