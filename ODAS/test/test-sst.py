import websocket
import json
import numpy as np
import cv2
import math

pos = []
color = (255, 0, 0)

def on_message(ws, message):
	global pos, category
	# This function will be called when a message is received
	data = json.loads(message)
	# print(data)
	pos = data['sound']

	img = np.zeros((800,800,3), np.uint8)
	for i in pos:
		angle_radians = math.radians(i['angle'])
		x, y = i['x'], i['y']
		cv2.circle(img, (x, y), int(30*i['activity']), color, -1)
		# cv2.circle(img, (i['x'], i['y']), int((1-abs(i['z'])/200)*30), color, -1)
		cv2.putText(img, str(i['dist']),
					(x + 30, y), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
	cv2.imshow('pu', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		exit(0)

def on_error(ws, error):
	print("Error:", error)

def on_open(ws):
	print("WebSocket opened")

if __name__ == "__main__":
	# WebSocket URL
	ws_url = "ws://127.0.0.1:11451"

	# Create WebSocket connection
	ws = websocket.WebSocketApp(ws_url,
								on_message = on_message,
								on_error = on_error)
	ws.on_open = on_open

	# Run WebSocket connection infinitely
	ws.run_forever()