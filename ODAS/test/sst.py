#!/usr/bin/env python
import socket
import sys
import threading
import random
import os
import time
import struct
import cv2
import signal
import json
import ast
import numpy as np
import traceback 

stop = False
HOST = "0.0.0.0"
PORT = 19001
SOCK_ADDR = (HOST, PORT)

def stop_handler(signum, frame):
	global running
	running = False

signal.signal(signal.SIGINT, stop_handler)
spectrum_rgb3_lut = [
	[ 0, 0, 0 ],
	[ 0, 0, 3 ],
	[ 0, 0, 6 ],
	[ 0, 0, 9 ],
	[ 0, 0, 12 ],
	[ 0, 0, 15 ],
	[ 0, 0, 18 ],
	[ 0, 0, 21 ],
	[ 0, 0, 24 ],
	[ 0, 0, 27 ],
	[ 0, 0, 30 ],
	[ 0, 0, 33 ],
	[ 0, 0, 36 ],
	[ 0, 0, 39 ],
	[ 0, 0, 42 ],
	[ 0, 0, 45 ],
	[ 0, 0, 48 ],
	[ 0, 0, 51 ],
	[ 0, 0, 54 ],
	[ 0, 0, 57 ],
	[ 0, 0, 60 ],
	[ 0, 0, 63 ],
	[ 0, 0, 66 ],
	[ 0, 0, 69 ],
	[ 0, 0, 72 ],
	[ 0, 0, 75 ],
	[ 0, 0, 78 ],
	[ 0, 0, 81 ],
	[ 0, 0, 84 ],
	[ 0, 0, 87 ],
	[ 0, 0, 90 ],
	[ 0, 0, 93 ],
	[ 0, 0, 96 ],
	[ 0, 0, 99 ],
	[ 0, 0, 102 ],
	[ 0, 0, 105 ],
	[ 0, 0, 108 ],
	[ 0, 0, 111 ],
	[ 0, 0, 114 ],
	[ 0, 0, 117 ],
	[ 0, 0, 120 ],
	[ 0, 0, 123 ],
	[ 0, 0, 126 ],
	[ 0, 0, 129 ],
	[ 0, 0, 132 ],
	[ 0, 0, 135 ],
	[ 0, 0, 138 ],
	[ 0, 0, 141 ],
	[ 0, 0, 144 ],
	[ 0, 0, 147 ],
	[ 0, 0, 150 ],
	[ 0, 0, 153 ],
	[ 0, 0, 156 ],
	[ 0, 0, 159 ],
	[ 0, 0, 162 ],
	[ 0, 0, 165 ],
	[ 0, 0, 168 ],
	[ 0, 0, 171 ],
	[ 0, 0, 174 ],
	[ 0, 0, 177 ],
	[ 0, 0, 180 ],
	[ 0, 0, 183 ],
	[ 0, 0, 186 ],
	[ 0, 0, 189 ],
	[ 0, 0, 192 ],
	[ 0, 0, 195 ],
	[ 0, 0, 198 ],
	[ 0, 0, 201 ],
	[ 0, 0, 204 ],
	[ 0, 0, 207 ],
	[ 0, 0, 210 ],
	[ 0, 0, 213 ],
	[ 0, 0, 216 ],
	[ 0, 0, 219 ],
	[ 0, 0, 222 ],
	[ 0, 0, 225 ],
	[ 0, 0, 228 ],
	[ 0, 0, 231 ],
	[ 0, 0, 234 ],
	[ 0, 0, 237 ],
	[ 0, 0, 240 ],
	[ 0, 0, 243 ],
	[ 0, 0, 246 ],
	[ 0, 0, 249 ],
	[ 0, 0, 252 ],
	[ 0, 0, 255 ],
	[ 0, 3, 252 ],
	[ 0, 6, 249 ],
	[ 0, 9, 246 ],
	[ 0, 12, 243 ],
	[ 0, 15, 240 ],
	[ 0, 18, 237 ],
	[ 0, 21, 234 ],
	[ 0, 24, 231 ],
	[ 0, 27, 228 ],
	[ 0, 30, 225 ],
	[ 0, 33, 222 ],
	[ 0, 36, 219 ],
	[ 0, 39, 216 ],
	[ 0, 42, 213 ],
	[ 0, 45, 210 ],
	[ 0, 48, 207 ],
	[ 0, 51, 204 ],
	[ 0, 54, 201 ],
	[ 0, 57, 198 ],
	[ 0, 60, 195 ],
	[ 0, 63, 192 ],
	[ 0, 66, 189 ],
	[ 0, 69, 186 ],
	[ 0, 72, 183 ],
	[ 0, 75, 180 ],
	[ 0, 78, 177 ],
	[ 0, 81, 174 ],
	[ 0, 84, 171 ],
	[ 0, 87, 168 ],
	[ 0, 90, 165 ],
	[ 0, 93, 162 ],
	[ 0, 96, 159 ],
	[ 0, 99, 156 ],
	[ 0, 102, 153 ],
	[ 0, 105, 150 ],
	[ 0, 108, 147 ],
	[ 0, 111, 144 ],
	[ 0, 114, 141 ],
	[ 0, 117, 138 ],
	[ 0, 120, 135 ],
	[ 0, 123, 132 ],
	[ 0, 126, 129 ],
	[ 0, 129, 126 ],
	[ 0, 132, 123 ],
	[ 0, 135, 120 ],
	[ 0, 138, 117 ],
	[ 0, 141, 114 ],
	[ 0, 144, 111 ],
	[ 0, 147, 108 ],
	[ 0, 150, 105 ],
	[ 0, 153, 102 ],
	[ 0, 156, 99 ],
	[ 0, 159, 96 ],
	[ 0, 162, 93 ],
	[ 0, 165, 90 ],
	[ 0, 168, 87 ],
	[ 0, 171, 84 ],
	[ 0, 174, 81 ],
	[ 0, 177, 78 ],
	[ 0, 180, 75 ],
	[ 0, 183, 72 ],
	[ 0, 186, 69 ],
	[ 0, 189, 66 ],
	[ 0, 192, 63 ],
	[ 0, 195, 60 ],
	[ 0, 198, 57 ],
	[ 0, 201, 54 ],
	[ 0, 204, 51 ],
	[ 0, 207, 48 ],
	[ 0, 210, 45 ],
	[ 0, 213, 42 ],
	[ 0, 216, 39 ],
	[ 0, 219, 36 ],
	[ 0, 222, 33 ],
	[ 0, 225, 30 ],
	[ 0, 228, 27 ],
	[ 0, 231, 24 ],
	[ 0, 234, 21 ],
	[ 0, 237, 18 ],
	[ 0, 240, 15 ],
	[ 0, 243, 12 ],
	[ 0, 246, 9 ],
	[ 0, 249, 6 ],
	[ 0, 252, 3 ],
	[ 0, 255, 0 ],
	[ 3, 252, 0 ],
	[ 6, 249, 0 ],
	[ 9, 246, 0 ],
	[ 12, 243, 0 ],
	[ 15, 240, 0 ],
	[ 18, 237, 0 ],
	[ 21, 234, 0 ],
	[ 24, 231, 0 ],
	[ 27, 228, 0 ],
	[ 30, 225, 0 ],
	[ 33, 222, 0 ],
	[ 36, 219, 0 ],
	[ 39, 216, 0 ],
	[ 42, 213, 0 ],
	[ 45, 210, 0 ],
	[ 48, 207, 0 ],
	[ 51, 204, 0 ],
	[ 54, 201, 0 ],
	[ 57, 198, 0 ],
	[ 60, 195, 0 ],
	[ 63, 192, 0 ],
	[ 66, 189, 0 ],
	[ 69, 186, 0 ],
	[ 72, 183, 0 ],
	[ 75, 180, 0 ],
	[ 78, 177, 0 ],
	[ 81, 174, 0 ],
	[ 84, 171, 0 ],
	[ 87, 168, 0 ],
	[ 90, 165, 0 ],
	[ 93, 162, 0 ],
	[ 96, 159, 0 ],
	[ 99, 156, 0 ],
	[ 102, 153, 0 ],
	[ 105, 150, 0 ],
	[ 108, 147, 0 ],
	[ 111, 144, 0 ],
	[ 114, 141, 0 ],
	[ 117, 138, 0 ],
	[ 120, 135, 0 ],
	[ 123, 132, 0 ],
	[ 126, 129, 0 ],
	[ 129, 126, 0 ],
	[ 132, 123, 0 ],
	[ 135, 120, 0 ],
	[ 138, 117, 0 ],
	[ 141, 114, 0 ],
	[ 144, 111, 0 ],
	[ 147, 108, 0 ],
	[ 150, 105, 0 ],
	[ 153, 102, 0 ],
	[ 156, 99, 0 ],
	[ 159, 96, 0 ],
	[ 162, 93, 0 ],
	[ 165, 90, 0 ],
	[ 168, 87, 0 ],
	[ 171, 84, 0 ],
	[ 174, 81, 0 ],
	[ 177, 78, 0 ],
	[ 180, 75, 0 ],
	[ 183, 72, 0 ],
	[ 186, 69, 0 ],
	[ 189, 66, 0 ],
	[ 192, 63, 0 ],
	[ 195, 60, 0 ],
	[ 198, 57, 0 ],
	[ 201, 54, 0 ],
	[ 204, 51, 0 ],
	[ 207, 48, 0 ],
	[ 210, 45, 0 ],
	[ 213, 42, 0 ],
	[ 216, 39, 0 ],
	[ 219, 36, 0 ],
	[ 222, 33, 0 ],
	[ 225, 30, 0 ],
	[ 228, 27, 0 ],
	[ 231, 24, 0 ],
	[ 234, 21, 0 ],
	[ 237, 18, 0 ],
	[ 240, 15, 0 ],
	[ 243, 12, 0 ],
	[ 246, 9, 0 ],
	[ 249, 6, 0 ],
	[ 252, 3, 0 ],
	[ 255, 0, 0 ]]

class SocketClientObject(object):
	def __init__(self, socket, address ):
		self.socket = socket
		self.address = address

echo_lock = threading.Lock()
now_pos = {}

class ClientThread(threading.Thread):
	def __init__(self, client_object):
		threading.Thread.__init__(self)
		self.client_object = client_object
	def run(self):
		global running, echo_lock, now_x, now_y, now_energy
		tmp_buffer = ''
		while running == True:
			now_buffer = self.client_object.socket.recv(1024).decode('UTF-8')

			tmp_buffer += now_buffer
			separated = tmp_buffer.split('\n}')
			detected = []
			for i in separated:
				try:
					received_json = json.loads(i + '\n}')
					if 'src' not in received_json:
						raise Exception
					detected.append(received_json)
				except Exception:
					tmp_buffer = i
					break
			for data in detected:
				try:
					target = data['src'][0]
					target_x = int(float(target["x"]) * 400)
					target_y = int(-float(target["y"]) * 400)
					energy = int(float(target["E"]) * 255)
					if (target_x in range(-40, 40) and target_y in range(-40, 40)) or energy < 100:
						continue
					with echo_lock:
						now_x, now_y, now_energy = target_x + 400, target_y + 400, energy
				except Exception as e:
					print(data)
					traceback.print_exc()
		self.client_object.socket.close()

class ClientThread2(threading.Thread):
	def __init__(self, client_object):
		threading.Thread.__init__(self)
		self.client_object = client_object
	def run(self):
		global running, echo_lock, now_x, now_y, now_energy
		tmp_buffer = ''
		while running == True:
			now_buffer = self.client_object.socket.recv(1024).decode('UTF-8')

			tmp_buffer += now_buffer
			separated = tmp_buffer.split('\n}')
			detected = []
			for i in separated:
				try:
					received_json = json.loads(i + '\n}')
					if 'src' not in received_json:
						raise Exception
					detected.append(received_json)
				except Exception:
					tmp_buffer = i
					break
			for data in detected:
				try:
					for target in data['src']:
						if target['id'] == 0:
							continue
						target_x = int(float(target["x"]) * 400)
						target_y = int(-float(target["y"]) * 400)
						# if (target_x in range(-40, 40) and target_y in range(-40, 40)):
						# 	continue
						with echo_lock:
							if target['id'] in now_pos and target['activity'] < 0.1:
								last_activity = now_pos[target['id']]['last_activity']
							else:
								last_activity = time.time()
							now_pos[target['id']] = {'x': target_x + 400, 'y': target_y + 400, 'z': target['z'], 'activity': target['activity'], 'last_activity': last_activity}
							#print(target, end='\r')
				except Exception as e:
					print(data)
					traceback.print_exc()
		self.client_object.socket.close()

class WindowThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
	def run(self):
		global running, echo_lock, now_x, now_y, now_energy
		while running == True:
			img = np.zeros((800,800,3),np.uint8)
			with echo_lock:
				try:
					for id_ in now_pos:
						if time.time() - now_pos[id_]['last_activity'] < 0.5:
							now_energy = 0
							cv2.circle(img, (now_pos[id_]['x'], now_pos[id_]['y']), 30, (spectrum_rgb3_lut[255- now_energy][0], spectrum_rgb3_lut[255- now_energy][1], spectrum_rgb3_lut[255- now_energy][2]), -1)
					cv2.imshow('pu', img)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				except Exception as e:
					traceback.print_exc()
					break
			time.sleep(0.02)
		cv2.destroyAllWindows()

def main():
	global running
	running = True
	try:
		sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock1.bind(SOCK_ADDR)
		sock1.listen(5)
		while running:
			(clientsocket, address) = sock1.accept()
			print(" Accept client: ", address)
			ct = ClientThread2(SocketClientObject(clientsocket, address))
			ct.start()
			wnd = WindowThread()
			wnd.start()
			print("Destination Connected by", address)
	except:
		print("#! EXC: ", sys.exc_info())
		sock1.close()
		sock2.close()
		print("THE END! Goodbye!")
if __name__ == "__main__":
	main()
