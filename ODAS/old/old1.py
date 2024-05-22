import sys
import time
import glob
import os
import pyaudio

import numpy as np
from openvino.inference_engine import IECore
import socket
import threading
import json

from tuning import Tuning
import usb.core
import usb.util
import struct
import math
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

import hashlib
import hmac
import base64
from websocket import create_connection
import websocket
from urllib.parse import quote
import logging
import re
import queue

import traceback

device = "CPU"
model_path = "public/aclnet-int8/aclnet_des_53_int8.xml"
label_path = "data/aclnet_53cl.txt"
whitelist = open('data/whitelist.txt', 'r').read().strip().split('\n')

microphone = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic_name = 'default'
mic_tuning = Tuning(microphone)

doa_lock = threading.Lock()
now_pos = {}

send_lock = threading.Lock()
app = FastAPI()

now_client = None
asr_running = False

def get_mic_index(name):
	p = pyaudio.PyAudio()
	devices = p.get_device_count()
	for i in range(devices):
		device_info = p.get_device_info_by_index(i)
		if device_info.get('maxInputChannels') > 0 and device_info.get('name') == name:
			return i

mic_index = get_mic_index(mic_name)
audio_lock = threading.Lock()
now_audio = queue.Queue()

def sound_classification(tcp_socket, client_socket, reg_num):
	global doa_lock, now_pos, audio_lock, now_audio, now_client
	now_client = client_socket
	ie = IECore()
	net = ie.read_network(model_path, model_path[:-4] + ".bin")
	input_blob = next(iter(net.input_info))
	input_shape = net.input_info[input_blob].input_data.shape
	output_blob = next(iter(net.outputs))
	exec_net = ie.load_network(network=net, device_name=device)
	labels = []
	if label_path:
		with open(label_path, "r") as file:
			labels = [line.rstrip() for line in file.readlines()]
	batch_size, channels, one, length = input_shape
	input_size = input_shape[-1]
	sample_rate = 16000
	audio = pyaudio.PyAudio()
	record_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input =True, frames_per_buffer=input_size, input_device_index=mic_index)

	chunk = np.zeros(input_shape, dtype=np.float32)
	hop = int(input_size * 0.8)
	overlap = int(input_size - hop)
	last = 'silence'

	while True:
		raw = record_stream.read(num_frames=hop)
		with audio_lock:
			now_audio.put(raw)

		input_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * 8.0
		scale = np.std(input_audio)
		scale = 4000 if scale < 4000 else scale	 # Limit scale value
		input_audio = (input_audio - np.mean(input_audio)) / scale
		input_audio = np.reshape(input_audio, ( 1, 1, 1, hop))

		chunk[:, :, :, :overlap] = chunk[:, :, :, -overlap:]
		chunk[:, :, :, overlap:] = input_audio
		
		output = exec_net.infer(inputs={input_blob: chunk})
		output = output[output_blob]
		for batch, data in enumerate(output):
			label = np.argmax(data)
			if data[label] < 0.8:
				label_txt = 'silence'
				illust_idx = 99
			else:
				label_txt = labels[label]
				illust_idx = label
		try:
			if label_txt != last:
				last = label_txt
				if label_txt != 'silence' and label_txt not in whitelist:
					continue
				result = ''
				if label_txt == 'silence':
					result = json.dumps({'type': 'SoundStopped'})
				else:
					with doa_lock:
						result = json.dumps({'type': 'SoundDetected', 'data': {'category': label_txt, 'acc': float(data[label])}, 'pos': [now_pos[i] for i in now_pos if time.time() - now_pos[i]['last_activity'] < 0.5]})

				result += '\n'
				with send_lock:
					try:
						client_socket.send(result.encode('UTF-8'))
						print(result, end='')
					except Exception:
						break
		except Exception as e:
			print(e)
			break
	print('Connection closed')
	record_stream.stop_stream()
	record_stream.close()

def doa_old(tcp_socket, client_socket, reg_num):
	last_angle = 0
	#last_voice_active = False
	while True:
		now_angle = mic_tuning.direction
		if now_angle != last_angle:
			last_angle = now_angle
			with send_lock:
				try:
					result = json.dumps({'type': 'SoundDirection', 'angle': 360-now_angle})
					result += '\n'
					client_socket.send(result.encode('UTF-8'))
					print(result, end='')
				except Exception as e:
					break
		time.sleep(0.2)

def doa_server():
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	server.bind(('0.0.0.0', 11450))
	server.listen(1)
	while True:
		(client_socket, address) = server.accept()
		print('DOA Client from {}'.format(address))
		threading.Thread(target=doa_handle, args=(client_socket,)).start()

def doa_handle(client_socket):
	global now_pos
	tmp_buffer = ''
	while True:
		try:
			now_buffer = client_socket.recv(1024).decode('UTF-8')
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
					for target in data['src']:
						if target['id'] == 0:
							continue
						x = int(float(target["x"]) * 400)
						y = int(-float(target["y"]) * 400)
						z = int(float(target["z"] - 0.5) * 400)
						with doa_lock:
							if target['id'] in now_pos and target['activity'] < 0.1:
								last_activity = now_pos[target['id']]['last_activity']
							else:
								last_activity = time.time()
							now_angle = math.degrees(math.atan2(y, x)) + 180
							now_dist = math.sqrt(x ** 2 + y ** 2)
							now_pos[target['id']] = {'x': x + 400, 'y': y + 400, 'z': z, 'activity': target['activity'], 'last_activity': last_activity, 'angle': now_angle, 'dist': now_dist}
		except Exception as e:
			print(traceback.format_exc())
			break

def doa(tcp_socket, client_socket, reg_num):
	global now_pos
	while True:
		pos = []
		with doa_lock:
			for id_ in now_pos:
				if time.time() - now_pos[id_]['last_activity'] < 0.5:
					pos.append(now_pos[id_])
			with send_lock:
				try:
					result = json.dumps({'type': 'SoundDirection', 'sound': pos})
					result += '\n'
					client_socket.send(result.encode('UTF-8'))
					print(result, end='')
				except Exception as e:
					break
		time.sleep(0.2)

class ASRClient():
	def __init__(self):
		self.app_id = "fb5a7a05"
		self.api_key = "d9de6a3c9059f1b8fb73a162d5e5689f"
		base_url = "ws://rtasr.xfyun.cn/v1/ws"
		ts = str(int(time.time()))
		tt = (self.app_id + ts).encode('utf-8')
		md5 = hashlib.md5()
		md5.update(tt)
		baseString = md5.hexdigest()
		baseString = bytes(baseString, encoding='utf-8')

		apiKey = self.api_key.encode('utf-8')
		signa = hmac.new(apiKey, baseString, hashlib.sha1).digest()
		signa = base64.b64encode(signa)
		signa = str(signa, 'utf-8')
		self.end_tag = "{\"end\": true}"

		self.ws = create_connection(base_url + "?appid=" + self.app_id + "&ts=" + ts + "&signa=" + quote(signa))
		print(self.ws.connected)
		self.trecv = threading.Thread(target=self.recv)
		self.trecv.start()

	def send(self):
		global now_audio, audio_lock
		print("- - - - - - - Start Recording ...- - - - - - - ")
		buffer = b''
		while True:
			if now_audio.empty():
				continue
			with audio_lock:
				chunk = now_audio.get()
			self.ws.send(chunk)
		print('ASR Send closed')

	def recv(self):
		global asr_running, now_client
		try:
			print('started')
			while asr_running and self.ws.connected:
				result = str(self.ws.recv())
				if len(result) == 0:
					print("receive result end")
					break
				result_dict = json.loads(result)
				# 解析结果
				if result_dict["action"] == "started":
					print("handshake success, result: " + result)

				if result_dict["action"] == "result":
					result = ''
					result_1 = re.findall('"w":"(.*?)"', str(result_dict["data"]))
					for i in result_1:
						if i == '。' or i == '.。' or i == ' .。' or i == ' 。':
							pass
						else:
							result += i
					print(result)
					now_client.sendall(json.dumps({'type': 'SpeechRecognition', 'text': result}).encode('UTF-8'))

				if result_dict["action"] == "error":
					print("rtasr error: " + result)
					self.ws.close()
					return
		except Exception as e:
			print(e)
			print("receive result end")
		print(asr_running)
		print(self.ws.connected)
		print('ASR Recv closed')

	def close(self):
		self.ws.close()
		print("connection closed")

now_asr = None

@app.get('/asr/start')
async def asr_start():
	global now_client, asr_running
	try:
		asr_running = True
		now_asr = ASRClient()
		threading.Thread(target=now_asr.send, args=()).start()
		return JSONResponse(content={'status': True})
	except Exception as e:
		return JSONResponse(content={'status': False, 'error': str(e)})

@app.get('/asr/stop')
async def asr_stop():
	global asr_running
	asr_running = False
	return JSONResponse(content={'status': True})

def main(port):
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	server.bind(('0.0.0.0', port))
	server.listen(128)
	while True:
		try:
			clientSocket, addr = server.accept()
			print('Accepted connection from {}'.format(addr))
			reg_num = 0
			scThread = threading.Thread(target = sound_classification, args=(server, clientSocket, reg_num))
			scThread.start()
			doaThread = threading.Thread(target = doa, args=(server, clientSocket, reg_num))
			doaThread.start()
		except Exception:
			break
	server.close()

def asr_server():
	uvicorn.run(app, host='0.0.0.0', port=11452)
 
if __name__ == '__main__':
	threading.Thread(target = doa_server).start()
	threading.Thread(target = asr_server).start()
	main(11451)


