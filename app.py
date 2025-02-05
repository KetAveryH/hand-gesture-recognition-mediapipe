#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refactored main application that integrates the Canvas class from canvas.py.
It handles gesture processing and delegates all OpenCV drawing/display to Canvas.
"""

import csv
import pyautogui
import threading
import copy
import argparse
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time
import json
import ctypes
import pdb

# Mouse event definitions for Windows
MOUSEEVENTF_LEFTDOWN = 0x0002  # Left button down
MOUSEEVENTF_LEFTUP   = 0x0004  # Left button up
MOUSEEVENTF_WHEEL    = 0x0800  # Mouse wheel

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Import the Canvas class (from canvas.py)
from canvas import Canvas

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def move_cursor(x: int, y: int, cap_width: int, cap_height: int, screen_width: int, screen_height: int, scaling_x: float, scaling_y: float):
    x_ratio, y_ratio = screen_width / cap_width, screen_height / cap_height
    x_margin, y_margin = (cap_width / scaling_x) / 2, (cap_height / scaling_y) / 2
    x = x - x_margin
    y = y - y_margin
    x = (x * x_ratio) * scaling_x
    y = (y * y_ratio) * scaling_y
    ctypes.windll.user32.SetCursorPos(int(x), int(y))

def click_and_hold():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def release_click():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def click():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.8,
                        help='min_detection_confidence')
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5,
                        help='min_tracking_confidence')
    args = parser.parse_args()
    return args

class GestureController():
    PINCH_THRESHOLD = 0.7
    GRAB_THRESHOLD = 0.20
    HOLD_THRESHOLD = 0.2  # seconds
    SCROLL_SENSITIVITY = 0.8
    BUFFER_FILE = "position.json"
    
    def __init__(self, cache_size):
        if cache_size < 1:
            raise ValueError("Cache must be at least of size 1")
        # For storing recent hand landmarks.
        self.hand_history = deque([None] * cache_size, maxlen=cache_size)
        self.curr_hand_data = [None, None]
        # Hand state machines for left/right
        self.hand_state = {"left": "idle", "right": "idle"}
        self.hand_press_time = {"left": 0.0, "right": 0.0}
        self.pinch_threshold = self.PINCH_THRESHOLD
        self.grab_threshold  = self.GRAB_THRESHOLD
        self.hold_threshold  = self.HOLD_THRESHOLD
        self.scroll_sensitivity = self.SCROLL_SENSITIVITY
        self.last_scroll_y = 0.0
        # For screen mapping
        self.screen_width, self.screen_height = pyautogui.size()
     
    def controller(self, event):
        data = {
            "left_hand": {
                "position": {"x": None, "y": None, "z": None},
                "gesture": "idle",
                "timestamp": time.time_ns()
            },
            "right_hand": {
                "position": {"x": None, "y": None, "z": None},
                "gesture": "idle",
                "timestamp": time.time_ns()
            }
        }
    
        for hand, handedness in zip(event.multi_hand_landmarks, event.multi_handedness):
            self.curr_hand_data = hand
            hand_type = handedness.classification[0].label  # "Left" or "Right"
            pinch = self.pinch_strength(4, 8)
            grab = self.grab_strength()
            palm_y = hand.landmark[0].y if hand.landmark else 0.0

            if hand_type == "Right":
                if hand.landmark:
                    palm_x = hand.landmark[0].x
                    palm_y = hand.landmark[0].y
                    self.move_cursor(palm_x, palm_y)
                    gesture = self.hand_state["right"]
                    data["right_hand"]["position"] = {"x": palm_x, "y": palm_y, "z": 0.0}
                    data["right_hand"]["gesture"] = gesture
                    data["right_hand"]["timestamp"] = time.time_ns()
                    self.update_right_hand_state(grab_strength=grab, pinch_strength=pinch, palm_y=palm_y)
            
            elif hand_type == "Left":
                if hand.landmark:
                    palm_x = hand.landmark[0].x
                    palm_y = hand.landmark[0].y
                    gesture = self.hand_state["left"]
                    data["left_hand"]["position"] = {"x": palm_x, "y": palm_y, "z": 0.0}
                    data["left_hand"]["gesture"] = gesture
                    data["left_hand"]["timestamp"] = time.time_ns()
                    self.update_left_hand_state(grab_strength=grab, pinch_strength=pinch, palm_y=palm_y)

            self.hand_history.append(hand)
        
        self.dump_json_data(data)
        self.send_hand_data(self.hand_state)
    
    def send_hand_data(self, hand_state):
        # Placeholder: implement sending hand state to a server if needed.
        pass
            
    def dump_json_data(self, data):
        try:
            with open(self.BUFFER_FILE, "w") as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing to {self.BUFFER_FILE}: {e}")

    def distance(self, landmark_1, landmark_2):
        return math.sqrt(
            (landmark_1.x - landmark_2.x) ** 2 +
            (landmark_1.y - landmark_2.y) ** 2 +
            (landmark_1.z - landmark_2.z) ** 2
        )

    def pinch_strength(self, idx1=4, idx2=8):
        if self.curr_hand_data is None or not self.curr_hand_data.landmark:
            return 0.0
        lm1 = self.curr_hand_data.landmark[idx1]
        lm2 = self.curr_hand_data.landmark[idx2]
        wrist = self.curr_hand_data.landmark[0]
        mid_mcp = self.curr_hand_data.landmark[9]
        pinch_dist = self.distance(lm1, lm2)
        ref_dist = self.distance(wrist, mid_mcp)
        if ref_dist == 0:
            return 0.0
        strength = 1.0 - (pinch_dist / ref_dist)
        return max(0.0, min(1.0, strength))

    def grab_strength(self):
        if self.curr_hand_data is None or not self.curr_hand_data.landmark:
            return 0.0
        tips = [4, 8, 12, 16, 20]
        values = []
        for tip_idx in tips:
            if tip_idx == 4:
                values.append(self.pinch_strength(4, 9))
            else:
                values.append(self.pinch_strength(tip_idx, 0))
        return sum(values) / len(values) if values else 0.0
    
    def update_right_hand_state(self, grab_strength, pinch_strength, palm_y):
        current_time = time.time()
        current_state = self.hand_state["right"]
        pinch_active = (pinch_strength >= self.pinch_threshold)
        grab_active  = (grab_strength >= self.grab_threshold)
        if pinch_active and pinch_strength > grab_strength:
            gesture = "pinch"
        elif grab_active:
            gesture = "grab"
        else:
            gesture = None

        if current_state == "idle":
            if gesture == "pinch":
                self.hand_state["right"] = "pinch-pressing"
                self.hand_press_time["right"] = current_time
            elif gesture == "grab":
                self.hand_state["right"] = "grab-pressing"
                self.hand_press_time["right"] = current_time
                self.last_scroll_y = palm_y
                print("Right hand: Grab start. Will scroll if you hold long enough.")
        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                if (current_time - self.hand_press_time["right"]) >= self.hold_threshold:
                    self.hand_state["right"] = "pinch-holding"
                    print("Right hand: pinch-hold started.")
                    self.press_down()
            else:
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed < self.hold_threshold:
                    print("Right hand: short pinch => single click.")
                    self.trigger_click_event()                              #
                self.hand_state["right"] = "idle"
        elif current_state == "pinch-holding":
            if gesture == "pinch":
                pass
            else:
                print("Right hand: pinch-hold ended.")
                self.press_up()
                self.hand_state["right"] = "idle"
        elif current_state == "grab-pressing":
            if gesture == "grab":
                if (current_time - self.hand_press_time["right"]) >= self.hold_threshold:
                    self.hand_state["right"] = "grab-holding"
                    self.last_scroll_y = palm_y
                    print("Right hand: grab-hold => begin scrolling mode.")
            else:
                print("Right hand: short grab => no action.")
                self.hand_state["right"] = "idle"
        elif current_state == "grab-holding":
            if gesture == "grab":
                self.scroll_with_displacement(palm_y)
            else:
                print("Right hand: grab scroll ended.")
                self.hand_state["right"] = "idle"

    def update_left_hand_state(self, grab_strength, pinch_strength, palm_y):
        current_time = time.time()
        current_state = self.hand_state["left"]
        pinch_active = (pinch_strength >= self.pinch_threshold)
        grab_active  = (grab_strength >= self.grab_threshold)
        if pinch_active and pinch_strength > grab_strength:
            gesture = "pinch"
        elif grab_active:
            gesture = "grab"
        else:
            gesture = None

        if current_state == "idle":
            if gesture == "pinch":
                self.hand_state["left"] = "pinch-pressing"
                self.hand_press_time["left"] = current_time
                print("Left hand pinch start.")
            elif gesture == "grab":
                self.hand_state["left"] = "grab-pressing"
                self.hand_press_time["left"] = current_time
                print("Left hand grab start - can do something else...")
        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                if (current_time - self.hand_press_time["left"]) >= self.hold_threshold:
                    self.hand_state["left"] = "pinch-holding"
                    print("Left hand pinch-hold started.")
            else:
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed < self.hold_threshold:
                    print("Left hand short pinch => single click.")
                    self.trigger_click_event()
                self.hand_state["left"] = "idle"
        elif current_state == "pinch-holding":
            if gesture == "pinch":
                pass
            else:
                print("Left hand pinch-hold ended.")
                self.hand_state["left"] = "idle"
        elif current_state == "grab-pressing":
            if gesture == "grab":
                if (current_time - self.hand_press_time["left"]) >= self.hold_threshold:
                    self.hand_state["left"] = "grab-holding"
                    print("Left hand grab-hold => do something (scroll, draw, etc).")
            else:
                print("Left hand short grab => no action.")
                self.hand_state["left"] = "idle"
        elif current_state == "grab-holding":
            if gesture == "grab":
                self.scroll_with_displacement(palm_y)
            else:
                print("Left hand grab ended.")
                self.hand_state["left"] = "idle"
    
    ### Mouse control functions
    ################ To Enable OS level controls simply comment out the following functions and commout out "pass". #########

    def scroll_with_displacement(self, current_y):
        # delta = current_y - self.last_scroll_y
        # scroll_amount = int(delta * self.scroll_sensitivity * 5000)
        # if scroll_amount != 0:
        #     ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, scroll_amount, 0)
        # self.last_scroll_y = current_y
        pass

    def move_cursor(self, x, y):
        # screen_x = int(x * self.screen_width)
        # screen_y = int(y * self.screen_height)
        # screen_x = max(0, min(self.screen_width - 1, screen_x))
        # screen_y = max(0, min(self.screen_height - 1, screen_y))
        # ctypes.windll.user32.SetCursorPos(screen_x, screen_y)
        pass

    def trigger_click_event(self):
        # self.press_down()
        # self.press_up()
        pass
    
    def press_down(self):
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        

    def press_up(self):
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:
        mode = 0
    if key == 107:
        mode = 1
    if key == 104:
        mode = 2
    return number, mode

def main():
    args = get_args()
    screen_width, screen_height = pyautogui.size()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_FPS, 60)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    hand_gesture_history = deque([[-1, -1]] * history_length, maxlen=history_length)
    drawn_circles = []
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    gesture_controller = GestureController(cache_size=10)
    canvas_instance = Canvas("Hand Gesture Recognition")

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC to exit
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        event = hands.process(image)
        image.flags.writeable = True

        if event.multi_hand_landmarks is not None:
            gesture_controller.controller(event)
        else:
            gesture_controller.dump_json_data(None)

        # Use the Canvas instance to draw all overlays and display the image.
        ########## UNCOMMENT THIS OUT IF DISPLAY IS NEEDED ###########################################
        # canvas_instance.display(debug_image, event, fps, mode, number, drawn_circles, list(point_history))  # introduced additional latency is introduced when canvas tab is minimized
                                                                           
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
