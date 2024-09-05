import sys
import time
import glob
import os
import pyaudio

import numpy as np
from openvino.inference_engine import IECore
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
import asyncio
import websockets
import subprocess
import socket
import psutil

from pixel_ring import pixel_ring

###################
#  Configuration  #
###################

device = 'CPU'

main_ws_port = 11451
odas_port = 11450

mic_name = 'default'
keep_after_seconds = 0.5

mic_color = 0x000000

model_path = "public/aclnet-int8/aclnet_des_53_int8.xml"
label_path = "data/aclnet_53cl.txt"

######################
#  Global Functions  #
######################

def get_mic_index(name):
	p = pyaudio.PyAudio()
	devices = p.get_device_count()
	for i in range(devices):
		device_info = p.get_device_info_by_index(i)
		if device_info.get('maxInputChannels') > 0 and device_info.get('name') == name:
			return i

def is_process_running(name):
	return any([i.info['name'] == name for i in psutil.process_iter(['pid', 'name', 'status'])])

######################
#  Initial settings  #
######################

if not is_process_running('pulseaudio'):
	os.system('pulseaudio -D')

pixel_ring.mono(mic_color)

app = FastAPI()

now_pos = {}
sst_lock = threading.Lock()

mic_index = get_mic_index(mic_name)

whitelist = open('data/whitelist.txt', 'r').read().strip().split('\n')
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

def handle_odas_server(client_socket):
	global now_pos
	last_buffer = ''
	while True:
		try:
			data = client_socket.recv(1024).decode('UTF-8')
		except Exception:
			print('Disconnected')
			break
		last_buffer += data
		separated = last_buffer.split('\n}')
		detected = []
		for stem in separated:
			try:
				received_json = json.loads(stem + '\n}')
				if 'src' not in received_json:
					raise Exception
				detected.append(received_json)
			except Exception:
				last_buffer = stem
				break
		with sst_lock:
			for data in detected:
				for target in data['src']:
					if target['id'] == 0:
						continue
					x, y, z = int(float(target["x"]) * 400), int(-float(target["y"]) * 400), int(float(target["z"] - 0.5) * 400)
					if target['id'] in now_pos and target['activity'] < 0.1:
						last_activity = now_pos[target['id']]['last_activity']
					else:
						last_activity = time.time()
					now_angle = math.degrees(math.atan2(y, x)) + 180
					now_dist = math.sqrt(x ** 2 + y ** 2)
					now_pos[target['id']] = {'id': target['id'], 'x': x + 400, 'y': y + 400, 'z': z, 'activity': target['activity'], 'last_activity': last_activity, 'angle': now_angle, 'dist': now_dist}
			now_pos = dict([(i, now_pos[i]) for i in now_pos if time.time() - now_pos[i]['last_activity'] < keep_after_seconds])

async def handle_doa(websocket):
	global now_pos
	while True:
		try:
			with sst_lock:
				result = json.dumps({'type': 'SoundDirection', 'sound': [now_pos[i] for i in now_pos]})
			await websocket.send(result)
		except websockets.exceptions.ConnectionClosedOK:
			print("Client connection closed.")
			break
		except websockets.exceptions.ConnectionClosedError:
			print("Client connection closed.")
			break
		await asyncio.sleep(0.2)

def handle_cls(websocket):
	global now_pos
	audio = pyaudio.PyAudio()
	record_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input =True, frames_per_buffer=input_size, input_device_index=mic_index)

	chunk = np.zeros(input_shape, dtype=np.float32)
	hop = int(input_size * 0.8)
	overlap = int(input_size - hop)
	last = 'silence'

	while True:
		raw = record_stream.read(num_frames=hop)

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
		if label_txt != last:
			last = label_txt
			if label_txt != 'silence' and label_txt not in whitelist:
				continue
			result = ''
			if label_txt == 'silence':
				result = json.dumps({'type': 'SoundStopped'})
			else:
				with sst_lock:
					result = json.dumps({'type': 'SoundDetected', 'data': {'category': label_txt, 'acc': float(data[label])}, 'sound': [now_pos[i] for i in now_pos]})
			try:
				asyncio.run(websocket.send(result))
			except websockets.exceptions.ConnectionClosedOK:
				print("Client connection closed.")
				break
	record_stream.stop_stream()
	record_stream.close()

def odas_server():
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	server.bind(('0.0.0.0', odas_port))
	server.listen(1)
	while True:
		(client_socket, address) = server.accept()
		print('DOA Client from {}'.format(address))
		threading.Thread(target=handle_odas_server, args=(client_socket,)).start()

async def handle_odas(websocket):
	try:
		server_task = asyncio.to_thread(odas_server)

		process = await asyncio.create_subprocess_exec(
			'/root/odaslive', '-c', '/root/config.cfg',
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE
		)

		await asyncio.gather(server_task, process.wait())
	except asyncio.CancelledError:
		server_task.cancel()
		process.terminate()


async def handle_ws(websocket, path):
	print("Client connected.")
	doa_task = asyncio.create_task(handle_doa(websocket))
	cls_task = asyncio.get_event_loop().run_in_executor(None, lambda: handle_cls(websocket))

	await asyncio.gather(doa_task, cls_task)

async def main():
	odas_task = asyncio.create_task(handle_odas(websocket))
	ws_server = await websockets.serve(handle_ws, '0.0.0.0', main_ws_port)

	await asyncio.gather(odas_task, ws_server.wait_closed())

asyncio.run(main())