import cv2
import mediapipe as mp
import landmark_utils as u
import asyncio
import websockets
import threading
import numpy as np
import time
import json
import copy
import csv
import itertools
from model import KeyPointClassifier
from PIL import Image, ImageDraw, ImageFont

DEBUG = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

now_image = None
image_lock = threading.Lock()

def predict(websocket):
    global now_image, image_lock

    number = 0
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            with image_lock:
                image = now_image
            if image == None:
                time.sleep(0.5)
                continue
            receivedKey = cv2.waitKey(20)
            number = receivedKey - 48
            try:
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(e)
                continue

            if results.multi_hand_landmarks and number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = u.pre_process_landmark(
                        landmark_list)
                    log_csv(number, pre_processed_landmark_list)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            final = image
            text = ""
            if number == -1:
                text = "Press key for gesture number"
            else:
                text = "Gesture: {}".format(number)
            cv2.putText(final, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', final)
            if cv2.waitKey(5) & 0xFF == 27:
                break

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def log_csv(number, landmark_list):
    if number > 9 or number == -1:
        pass
    csv_path = csv_path = 'model/keypoint_classifier/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return


async def receive(websocket):
    global now_image, image_lock
    while True:
        try:
            raw = await websocket.recv()
            length = int.from_bytes(raw[:4], byteorder='big')
            data = raw[4:]
            while len(data) < length:
                buf = await websocket.recv()
                data += buf
        except Exception as e:
            print('Disconnected')
            break
        with image_lock:
            now_image = data


async def handle_ws(websocket, path):
    print("Client connected.")
    loop = asyncio.get_event_loop()
    image_receive_task = asyncio.create_task(receive(websocket))
    image_predict_task = loop.run_in_executor(None, lambda: predict(websocket))

    await asyncio.gather(image_receive_task, image_predict_task)


async def main():
    ws_server = await websockets.serve(handle_ws, '0.0.0.0', 11454)

    await asyncio.gather(ws_server.wait_closed())

if __name__ == '__main__':
    asyncio.run(main())