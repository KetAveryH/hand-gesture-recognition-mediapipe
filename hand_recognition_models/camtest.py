import cv2 as cv
import time
import pdb

def main():
    # Initialize the video capture with the default device (ID 0)
    pdb.set_trace()
    cap = cv.VideoCapture(1)
    
    # Set resolution (optional)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    pdb.set_trace()
    cap.set(cv.CAP_PROP_FPS, 60)  # Set the FPS to 60, if supported
    
    # Initialize FPS calculation
    prev_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally
        frame = cv.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display FPS on the screen
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Display the frame
        cv.imshow('Camera Test', frame)

        # Exit on 'ESC' key
        if cv.waitKey(1) & 0xFF == 27:
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
