import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Smooth pointer
prev_x, prev_y = 0, 0
smoothening = 6

# Swipe buffer
swipe_points = deque(maxlen=10)
gesture_cooldown = 1
last_gesture_time = time.time()

drag_mode = False
click_hold_start = None
scroll_mode = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lm = handLms.landmark

        # Get fingertips
        index_tip = lm[8]
        thumb_tip = lm[4]
        middle_tip = lm[12]
        ring_tip = lm[16]

        # Convert to screen coordinates
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        screen_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
        screen_y = np.interp(index_tip.y, [0, 1], [0, screen_height])
        cur_x = prev_x + (screen_x - prev_x) / smoothening
        cur_y = prev_y + (screen_y - prev_y) / smoothening
        prev_x, prev_y = cur_x, cur_y

        # Move mouse 
        pyautogui.moveTo(cur_x, cur_y)

        # Measure distances
        def distance(a, b):
            return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

        pinch_dist = distance(index_tip, thumb_tip)
        middle_pinch_dist = distance(middle_tip, thumb_tip)
        index_middle_dist = distance(index_tip, middle_tip)

        # 1. Left Click
        if pinch_dist < 0.03:
            if not drag_mode:
                pyautogui.click()
                cv2.putText(img, 'Left Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(0.2)

        # 2. Right Click (thumb + middle)
        elif middle_pinch_dist < 0.03:
            pyautogui.rightClick()
            cv2.putText(img, 'Right Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            time.sleep(0.2)

        # 3. Drag Mode
        elif pinch_dist < 0.05:
            if not drag_mode:
                pyautogui.mouseDown()
                drag_mode = True
                cv2.putText(img, 'Dragging...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            if drag_mode:
                pyautogui.mouseUp()
                drag_mode = False
                cv2.putText(img, 'Dropped', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # 4. Scroll (index + middle fingers close)
        if index_middle_dist < 0.04:
            scroll_mode = True
            scroll_dir = prev_y - cur_y
            pyautogui.scroll(int(scroll_dir * 3))
            cv2.putText(img, 'Scrolling...', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            scroll_mode = False

        # 5. Three Finger Swipe
        if all([
            lm[8].y < lm[6].y,  # Index finger up
            lm[12].y < lm[10].y,  # Middle finger up
            lm[16].y < lm[14].y  # Ring finger up
        ]):
            swipe_points.append((x, y))
            if len(swipe_points) >= 5:
                dx = swipe_points[-1][0] - swipe_points[0][0]
                if abs(dx) > 80 and (time.time() - last_gesture_time > gesture_cooldown):
                    if dx > 0:
                        pyautogui.hotkey('ctrl', 'tab')
                        cv2.putText(img, 'Swipe Right (Next Tab)', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        pyautogui.hotkey('ctrl', 'shift', 'tab')
                        cv2.putText(img, 'Swipe Left (Prev Tab)', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    last_gesture_time = time.time()

        # Draw landmarks
        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
