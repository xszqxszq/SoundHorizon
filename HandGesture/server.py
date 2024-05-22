import cv2
import mediapipe as mp
from model import KeyPointClassifier
import landmark_utils as u
import asyncio
import websockets
import numpy as np
import time
import json
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageDraw, ImageFont

DEBUG = False

def get_device_by_name(name):
	devices = FilterGraph().get_input_devices()

	for device_index, data in enumerate(devices):
		if data[0] == name:
			return device_index
	return None

cap = cv2.VideoCapture(2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

kpclf = KeyPointClassifier()
font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8")

def drawText(img, x, y, string):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((x, y - 50), string, (255, 255, 255), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

gestures = {
	0: '你',
	1: '你',
	2: '我',
	3: '教',
	4: '手语',
	5: '认识',
	6: '', # 很高兴
	7: '',
	8: '',
}
single_hand_gestures = [1, 2]
multi_hands_gestures = [3, 4, 5, 6]


class HandGestureTask:
	def __init__(self, websocket):
		self.cnt = dict([(i, 0) for i in gestures])
		self.now_image = None
		self.image_lock = asyncio.Lock()
		self.detect_lock = asyncio.Lock()
		self.websocket = websocket
		self.title = 'Connection from {}:{}'.format(websocket.remote_address[0], websocket.remote_address[1])
		print(self.title)

	def infer(self, hands, image):
		# image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = hands.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		gesture_index = len(gestures) - 1

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				landmark_list = u.calc_landmark_list(image, hand_landmarks)
				keypoints = u.pre_process_landmark(landmark_list)
				try:
					gesture_index = kpclf(keypoints)
				except Exception:
					continue

				if DEBUG:
					mp_drawing.draw_landmarks(
						image,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
		if results.multi_handedness:
			if (gesture_index in multi_hands_gestures and len(results.multi_handedness) != 2) or (gesture_index in single_hand_gestures and len(results.multi_handedness) != 1):
				return len(gestures) - 1

		return gesture_index

	def debug_show(self, image, gesture_index):
		final = drawText(image, 10, 60, gestures[gesture_index])
		cv2.imshow(self.title, final)
		cv2.waitKey(5)

	async def get_hands(self, loop):
		return await loop.run_in_executor(None, lambda: mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5))

	async def start_inference(self, loop):
		hands = await self.get_hands(loop)
		while True:
			async with self.image_lock:
				image = self.now_image
			if not isinstance(image, np.ndarray):
				await asyncio.sleep(0.5)
				continue
			try:
				gesture_index = await loop.run_in_executor(None, lambda: self.infer(hands, image))
			except Exception as e:
				print(e)
				return None
			async with self.detect_lock:
				self.cnt[gesture_index] += 1
			if DEBUG:
				await loop.run_in_executor(None, lambda: self.debug_show(image, gesture_index))
		if DEBUG:
			cv2.destroyWindow(self.title)

	async def send(self):
		last = ''
		while True:
			await asyncio.sleep(1.0)
			async with self.detect_lock:
				if sum(self.cnt.values()) == 0:
					continue
				now = max(self.cnt, key=self.cnt.get)
				self.cnt = dict([(i, 0) for i in gestures])
			if last != now and now != '':
				res = json.dumps({'type': gestures[now]}, ensure_ascii=False)
				print(res)
				try:
					await self.websocket.send(res)
				except Exception as e:
					break
			last = now

	# async def receive(self):
	# 	while True:
	# 		try:
	# 			raw = await self.websocket.recv()
	# 			length = int.from_bytes(raw[:4], byteorder='big')
	# 			data = raw[4:]
	# 			while len(data) < length:
	# 				buf = await self.websocket.recv()
	# 				data += buf
	# 		except Exception as e:
	# 			print(e)
	# 			print('Disconnected')
	# 			break
	# 		async with self.image_lock:
	# 			self.now_image = data
	async def receive(self, loop):
		# TODO: Automatically choose OBS 1
		while True:
			success, image = await loop.run_in_executor(None, lambda: cap.read())
			if not success:
				continue
			async with self.image_lock:
				self.now_image = image
			await asyncio.sleep(1 / 24)

	async def handle(self):
		loop = asyncio.get_event_loop()
		image_receive_task = asyncio.create_task(self.receive(loop))
		image_predict_task = asyncio.create_task(self.start_inference(loop))
		image_send_task = asyncio.create_task(self.send())

		await asyncio.gather(image_receive_task, image_predict_task, image_send_task)


async def handle_ws(websocket, path):
	task = HandGestureTask(websocket)
	await task.handle()


async def main():
	ws_server = await websockets.serve(handle_ws, '0.0.0.0', 11454)
	print('Hand Gesture Server started.')

	await asyncio.gather(ws_server.wait_closed())

if __name__ == '__main__':
	asyncio.run(main())