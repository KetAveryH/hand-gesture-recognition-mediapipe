#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import pyautogui
import threading
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time

MOUSEEVENTF_LEFTDOWN = 0x0002  # Left button down
MOUSEEVENTF_LEFTUP   = 0x0004  # Left button up
MOUSEEVENTF_WHEEL    = 0x0800  # Mouse wheel

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# FOR WINDOWS TODO: Add if windows
import ctypes

import pdb

MOUSEEVENTF_LEFTDOWN = 0x0002  # Left button down
MOUSEEVENTF_LEFTUP = 0x0004    # Left button up

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def move_cursor(x: int, y: int, cap_width: int , cap_height: int, screen_width: int, screen_height: int, scaling_x: float, scaling_y):
    x_ratio, y_ratio = (screen_width/(cap_width)), (screen_height/(cap_height)) # x * x_ratio = screen_x_pos
    
    
    # Create Play Area box in center of video of dimension: cap_dimensions / scaling_sensitivity
    x_margin, y_margin = (cap_width/scaling_x) / 2, (cap_height/scaling_y) / 2  # Calculate total margin area, divide by 2
    x_min, y_min = x_margin, y_margin
    x_max, y_max = cap_width - x_margin, cap_height - y_margin 
    
    # Shift Play Area box towards the right by the margin
    x = x-x_margin
    y = y-y_margin
    
    # Clamp values to exist within the desktop borders
    # x = clamp(x, x_min, x_max)
    # y = clamp(y, y_min, y_max)
    
    x = (x * x_ratio) * scaling_x # Map x video pos to screen pos, multiply by scaling sensitivity 
    y = (y * y_ratio) * scaling_y  # Map y video to to screen pos. multiply by scaling sensitivity
    
    # Move Cursor
    ctypes.windll.user32.SetCursorPos(int(x), int(y))

def click_and_hold():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)  # Simulate left button press

def release_click():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)    # Simulate left button release

def click():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)  # Simulate left button press
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)    # Simulate left button release


def get_args():
    parser = argparse.ArgumentParser()
    
    

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.8)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

class GestureController():
    def __init__(self, cache_size):
        if cache_size < 1:
            raise ValueError("Cache must be at least of size 1")
        
        # For storing landmarks from recent frames.
        self.hand_history = deque([None] * cache_size, maxlen=cache_size)
        self.curr_hand_data = None
        
        # Simple hand state machines
        self.hand_state = {"left": "idle", "right": "idle"}
        self.hand_press_time = {"left": 0.0, "right": 0.0}

        # Pinch / grab thresholds
        self.pinch_threshold = 0.7
        self.grab_threshold  = 0.20
        self.hold_threshold  = 0.2  # seconds

        # Grab-to-scroll
        self.scroll_sensitivity = 0.8
        self.last_scroll_y = 0.0

        # For screen mapping
        import pyautogui
        self.screen_width, self.screen_height = pyautogui.size()
        
        
        # How do we want to store the history of hand detections?
        # We can assume we will only receive 2 detections at once
        # We will be able to get both of these detection "simultaneously" (unlike ultraleap)
        # We may want to store a series of n previous detections, stored together along with a timestamp
        # We can decide on a list for now of pairs, we can use a deque (double ended queue) data object to store our "cache"
     
    def controller(self, event):
        """
        Process the current Mediapipe detection (event.multi_hand_landmarks)
        and update hand states (pinch vs grab) for left and right hands.
        Also demonstrates how to move the cursor from the index-finger tip.
        """
        # For each detected hand + label (Left/Right)
        for hand, handedness in zip(event.multi_hand_landmarks,
                                    event.multi_handedness):
            self.curr_hand_data = hand
            hand_type = handedness.classification[0].label  # "Left" or "Right"

            # Calculate pinch and grab strengths
            pinch = self.pinch_strength(4, 8)  # thumb tip & index tip
            grab  = self.grab_strength()

            # We'll define the "palm_y" from WRIST index=0
            # (You can pick a different reference if you prefer)
            palm_y = hand.landmark[0].y if hand.landmark else 0.0

            if hand_type == "Right":
                # Move cursor from wrist to avoid instability (landmark 0)
                if hand.landmark:
                    pointer_x = hand.landmark[0].x
                    pointer_y = hand.landmark[0].y
                    self.move_cursor(pointer_x, pointer_y)

                # Update state machine
                self.update_right_hand_state(
                    grab_strength=grab,
                    pinch_strength=pinch,
                    palm_y=palm_y
                )
            
            elif hand_type == "Left":
                # Example: no cursor for left, or something else:
                self.update_left_hand_state(
                    grab_strength=grab,
                    pinch_strength=pinch,
                    palm_y=palm_y
                )

            # Keep some kind of history (make this left vs right for the future, and ID based)
            self.hand_history.append(hand)

    # ---------------------------
    # Pinch/Grab Strength Helpers
    # ---------------------------
    def distance(self, landmark_1, landmark_2):
        """
        Euclidean distance between two 3D points (x,y,z).
        """
        return math.sqrt(
            (landmark_1.x - landmark_2.x) ** 2 +
            (landmark_1.y - landmark_2.y) ** 2 +
            (landmark_1.z - landmark_2.z) ** 2
        )

    def pinch_strength(self, idx1=4, idx2=8):
        """
        Returns a normalized pinch strength between 0..1
        (0 = fully open, 1 = fully pinched).
        Default compares THUMB_TIP=4 and INDEX_TIP=8.
        """
        if self.curr_hand_data is None or not self.curr_hand_data.landmark:
            return 0.0

        # Landmark references
        lm1 = self.curr_hand_data.landmark[idx1]
        lm2 = self.curr_hand_data.landmark[idx2]
        wrist = self.curr_hand_data.landmark[0]  # WRIST
        mid_mcp = self.curr_hand_data.landmark[9]  # MIDDLE_FINGER_MCP for scaling

        # Distance between pinch landmarks
        pinch_dist = self.distance(lm1, lm2)
        # Reference distance
        ref_dist = self.distance(wrist, mid_mcp)
        if ref_dist == 0:
            return 0.0
        
        strength = 1.0 - (pinch_dist / ref_dist)
        return max(0.0, min(1.0, strength))

    def grab_strength(self):
        """
        Very rough 'grab' measure. 
        Averages multiple pinch strengths to see if the fingers are close to the wrist.
        """
        if self.curr_hand_data is None or not self.curr_hand_data.landmark:
            return 0.0
        
        # We can just re-use pinch_strength with various pairs:
        #  - Compare each fingertip to the wrist or to the palm center
        tips = [4, 8, 12, 16, 20]  # thumb/index/middle/ring/pinky tip
        values = []
        for tip_idx in tips:
            # Compare tip to wrist (0)
            # We'll do 1 - (distance(tip, wrist)/someRef). Already done in pinch_strength logic, so:
            # Let's just do pinch_strength(tip, 0) if you like
            if tip_idx == 4:
                # we can do a bigger difference for thumb to a bigger reference
                values.append(self.pinch_strength(4, 9))  # thumb to middle MCP
            else:
                # Compare fingertip to wrist
                values.append(self.pinch_strength(tip_idx, 0))

        # Average
        if values:
            return sum(values)/len(values)
        else:
            return 0.0
    
    # ---------------------------
    # Right-Hand State Machine
    # ---------------------------
    def update_right_hand_state(self, grab_strength, pinch_strength, palm_y):
        """
        Right-hand FSM logic:
        - Pinch => short pinch click, pinch-hold => click & hold
        - Grab => short grab => do nothing, grab-hold => scroll
        Priority: pinch if (pinch_strength >= pinch_threshold && pinch > grab);
                  else if (grab_strength >= grab_threshold) => grab
                  else idle
        """
        current_time = time.time()
        current_state = self.hand_state["right"]

        pinch_active = (pinch_strength >= self.pinch_threshold)
        grab_active  = (grab_strength >= self.grab_threshold)

        # Priority check
        if pinch_active and pinch_strength > grab_strength:
            gesture = "pinch"
        elif grab_active:
            gesture = "grab"
        else:
            gesture = None

        # State transitions
        if current_state == "idle":
            if gesture == "pinch":
                self.hand_state["right"] = "pinch-pressing"
                self.hand_press_time["right"] = current_time
                # Immediately we can do a press_down, or wait. 
                # self.press_down()
                
            elif gesture == "grab":
                self.hand_state["right"] = "grab-pressing"
                self.hand_press_time["right"] = current_time
                self.last_scroll_y = palm_y
                print("Right hand: Grab start. Will scroll if you hold long enough.")

        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                # Still pinching => check for hold threshold
                if (current_time - self.hand_press_time["right"]) >= self.hold_threshold:
                    self.hand_state["right"] = "pinch-holding"
                    print("Right hand: pinch-hold started.")
                    self.press_down()
            else:
                # If pinch ended before hold threshold => short pinch => click
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed < self.hold_threshold:
                    print("Right hand: short pinch => single click.")
                    self.trigger_click_event()
                self.hand_state["right"] = "idle"

        elif current_state == "pinch-holding":
            if gesture == "pinch":
                # Keep holding
                pass
            else:
                # pinch ended => release
                print("Right hand: pinch-hold ended.")
                self.press_up()
                self.hand_state["right"] = "idle"

        elif current_state == "grab-pressing":
            if gesture == "grab":
                # check for hold threshold => start scrolling
                if (current_time - self.hand_press_time["right"]) >= self.hold_threshold:
                    self.hand_state["right"] = "grab-holding"
                    self.last_scroll_y = palm_y
                    print("Right hand: grab-hold => begin scrolling mode.")
            else:
                # short grab => do nothing
                print("Right hand: short grab => no action.")
                self.hand_state["right"] = "idle"

        elif current_state == "grab-holding":
            if gesture == "grab":
                # keep scrolling
                self.scroll_with_displacement(palm_y)
            else:
                # done
                print("Right hand: grab scroll ended.")
                self.hand_state["right"] = "idle"

    # ---------------------------
    # Left-Hand State Machine
    # ---------------------------
    def update_left_hand_state(self, grab_strength, pinch_strength, palm_y):
        """
        Example left-hand FSM. 
        You can adapt the logic for drawing, clearing, or anything else you want.
        Currently, it parallels the right-hand approach.
        """
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

        # For demonstration, let's just do a simpler pinch click, or
        # you can replicate exactly the same logic as right-hand:
        if current_state == "idle":
            if gesture == "pinch":
                self.hand_state["left"] = "pinch-pressing"
                self.hand_press_time["left"] = current_time
                # e.g., you might do self.press_down() or start drawing
                print("Left hand pinch start.")

            elif gesture == "grab":
                self.hand_state["left"] = "grab-pressing"
                self.hand_press_time["left"] = current_time
                print("Left hand grab start - can do something else...")

        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                # Still pinching => check for hold threshold
                if (current_time - self.hand_press_time["left"]) >= self.hold_threshold:
                    self.hand_state["left"] = "pinch-holding"
                    print("Left hand pinch-hold started.")
            else:
                # short pinch => single click
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed < self.hold_threshold:
                    print("Left hand short pinch => single click.")
                    self.trigger_click_event()
                self.hand_state["left"] = "idle"

        elif current_state == "pinch-holding":
            if gesture == "pinch":
                # keep pinch-holding
                pass
            else:
                # pinch ended => release
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
                # If you want left-hand to scroll, you can do:
                self.scroll_with_displacement(palm_y)
            else:
                print("Left hand grab ended.")
                self.hand_state["left"] = "idle"

    # ---------------------------
    # Scrolling + Mouse Helpers
    # ---------------------------
    def scroll_with_displacement(self, current_y):
        """
        Compare current_y with last_scroll_y, 
        send mouse wheel event for difference.
        Positive => scroll up, negative => scroll down (by default).
        """
        delta = current_y - self.last_scroll_y
        scroll_amount = int( delta * self.scroll_sensitivity * 5000)

        if scroll_amount != 0:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, scroll_amount, 0)
        
        self.last_scroll_y = current_y

    def move_cursor(self, x, y):
        """
        Convert normalized [0..1] x,y from Mediapipe 
        to actual screen coordinates and move the OS cursor.
        """
        # Might need to invert y since Mediapipe y=0 top, 
        # and Windows y=0 top. Usually flipping is not needed if you want direct movement, 
        # but you can invert if you like:
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)

        # Clamp if needed
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        ctypes.windll.user32.SetCursorPos(screen_x, screen_y)

    def trigger_click_event(self):
        self.press_down()
        self.press_up()

    def press_down(self):
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def press_up(self):
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
    
                

def main():
    # Argument parsing #################################################################
    args = get_args()

    screen_width, screen_height = pyautogui.size()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # pdb.set_trace()
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_FPS, 60)


    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Hand gesture history 
    hand_gesture_history = deque([[-1,-1]]*history_length, maxlen=history_length) #deque([left_gesture, right_gesture], ...)
    drawn_circles = []  # Store drawn circles
    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    
    # GestureController
    gesture_controller = GestureController(cache_size=10)

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        event = hands.process(image)
        image.flags.writeable = True

        ####################################################################
        if event.multi_hand_landmarks is not None:
            # Gesture Controller
                # Will take in all current landmark detections and process them
            
            gesture_controller.controller(event)
            
            for hand_landmarks, handedness in zip(event.multi_hand_landmarks,
                                                  event.multi_handedness):
                hand_side = handedness.classification[0].label # returns: "Left" or "Right"
                
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                
                
                #
                if hand_side == "Left":
                    hand_gesture_history.append([hand_sign_id, None])
                else:
                    hand_gesture_history.append([None, hand_sign_id])
                    
                    
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                # Gesture control
                pointer_x, pointer_y = landmark_list[8]
                if hand_side == "Right" and hand_sign_id == 2: # Right hand is pointing  # TODO: Program so it adapts for lefties and righties
                    
                    
                    move_cursor(pointer_x, pointer_y, cap_width, cap_height, screen_width, screen_height, 2, 1.5)
                    
                    if hand_gesture_history[-2][0] == 1: # previous frame of left hand is closed
                        # Draw circle permanently here
                        
                        drawn_circles.append((pointer_x, pointer_y))
                        # Hold the click
                        click()
                        

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                debug_image = draw_drawing(debug_image, drawn_circles)
    
                

                
                        
                        
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        debug_image = draw_drawing(debug_image, drawn_circles)
       
        
        

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_drawing(image, drawn_circles):
    if len(drawn_circles) > 0 :
        for circle in drawn_circles:
            cv.circle(image, (int(circle[0]), int(circle[1])), 5, (0, 255, 0), -1)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
