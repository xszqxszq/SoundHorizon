import aiohttp
import asyncio
import json
import math
import netifaces
import numpy as np
import os
import psutil
import pyaudio
import socket
import struct
import subprocess
import time
import traceback
import websockets
import winreg
import yaml
import sounddevice as sd
from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from netifaces import AF_INET
from openvino.inference_engine import IECore


p = pyaudio.PyAudio()
stream = p.open(
	format=pyaudio.paInt16,
	channels=1,
	rate=16000,
	output=True,
	frames_per_buffer=1024
)

async def extract_json(buf):
	separated = buf.split('\n}')
	detected = []
	for stem in separated:
		try:
			received_json = json.loads(stem + '\n}')
			if 'src' not in received_json:
				raise Exception
			detected.append(received_json)
		except Exception:
			buf = stem
			break
	return buf, detected

def get_connection_name_from_guid():
	iface_guids = netifaces.interfaces()
	iface_names = dict([(iface_guids[i], '(unknown)') for i in range(len(iface_guids))])
	reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
	reg_key = winreg.OpenKey(reg, r'SYSTEM\CurrentControlSet\Control\Network\{4d36e972-e325-11ce-bfc1-08002be10318}')
	for iface in iface_guids:
		try:
			reg_subkey = winreg.OpenKey(reg_key, iface + r'\Connection')
			iface_names[iface] = winreg.QueryValueEx(reg_subkey, 'Name')[0]
		except FileNotFoundError:
			pass
	return {v: k for k, v in iface_names.items()}

def get_mic_index(name):
	p = pyaudio.PyAudio()
	devices = p.get_device_count()
	for i in range(devices):
		device_info = p.get_device_info_by_index(i)
		if device_info.get('maxInputChannels') > 0 and device_info.get('name') == name:
			return i

def load_config():
	with open('config.yml', 'r', encoding='UTF-8') as f:
		return yaml.safe_load(f)

class CLS:
	def __init__(self):
		self.config = config['cls']
		self.config_audio = config['sst']['audio']
		with open(self.config['whitelist'], 'r') as f:
			self.whitelist = f.read().strip().split('\n')
		self.max_sounds = config['sst']['max_sounds']
		self.fps = self.config_audio['fps']

		self.plt_classes = [0, 76, 106, 302, 348, 353, 494]
		self.window = 0.975
		self.sample_rate = 16000
		self.chunk = int(self.window * self.sample_rate)
		self.model = YAMNet(weights='keras_yamnet/yamnet.h5')
		self.yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')
		self.n_classes = len(self.plt_classes)

		self.lock = asyncio.Lock()
		self.audio = [None for i in range(self.max_sounds)]
		self.label, self.acc = ['silence' for i in range(self.max_sounds)], [0.0 for i in range(self.max_sounds)]

	def update_audio(self, raw):
		for i in range(self.max_sounds):
			if len(raw[i]) == 0:
				return False
			try:
				self.audio[i] = raw[i]
			except Exception as e:
				print(e)
				return False
		return True

	async def read_chunk(self, loop):
		while True:
			async with self.sst.frame_lock:
				raw = await self.sst.get_frame_async(loop)
				if isinstance(raw[0], np.ndarray):
					break
			await asyncio.sleep(1 / self.fps)
		return await loop.run_in_executor(None, self.update_audio, raw)

	def infer(self):
		label, acc = ['silence' for i in range(self.max_sounds)], [0.0 for i in range(self.max_sounds)]
		for i in range(self.max_sounds):
			if not isinstance(self.audio[i], np.ndarray):
				continue
			data = preprocess_input(self.audio[i], self.sample_rate)
			prediction = self.model.predict(np.expand_dims(data, 0), verbose = 0)[0]

			ind = np.argmax(prediction[self.plt_classes])
			label[i], acc[i] = self.yamnet_classes[self.plt_classes][ind], float(prediction[self.plt_classes][ind])
		return label, acc

	async def infer_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.infer())

	async def handle(self, sst, ssl, loop):
		last = ['silence' for i in range(self.max_sounds)]
		self.sst = sst

		while True:
			if not (await self.read_chunk(loop)):
				continue
			label, acc = await self.infer_async(loop)
			async with self.lock:
				self.label, self.acc = label, acc

class SST:
	def __init__(self):
		self.config = config['sst']
		self.host = self.config['host']
		self.port = self.config['port']
		self.max_sounds = self.config['max_sounds']
		self.offset = self.config['offset']
		self.min_z, self.max_z = self.config['min_z'], self.config['max_z']
		self.port_audio = self.config['audio']['port']
		self.hop = self.config['audio']['hop']
		self.ttl = self.config['activity']['ttl']
		self.activity_threshold = self.config['activity']['threshold']

		self.lock = asyncio.Lock()
		self.pos = {}
		self.frame_lock = asyncio.Lock()
		self.waves = [None for i in range(self.max_sounds)]
		self.frame = [None for i in range(self.max_sounds)]

	async def clean_inactive(self):
		self.pos = dict([(i, self.pos[i]) for i in self.pos if time.time() - self.pos[i]['last_activity'] < self.ttl])

	async def update(self, target, ind):
		x, y, z = int(float(target['x']) * 400), int(-float(target['y']) * 400), abs(target['z'] - 0.5)
		if target['id'] in self.pos and target['activity'] < self.activity_threshold:
			last_activity = self.pos[target['id']]['last_activity']
		else:
			last_activity = time.time()
		angle, dist = (360 + math.degrees(math.atan2(y, x)) + self.offset) % 360, math.sqrt((x - 200) ** 2 + (y - 200) ** 2)
		self.pos[target['id']] = {
			'id': target['id'], 
			'x': x + 400, 'y': y + 400, 
			'z': z, 'activity': target['activity'],
			'last_activity': last_activity,  
			'h': target['z'], 
			'angle': angle, 
			'order': ind,
			'dist': dist
		}

	async def handle(self, client_socket, loop):
		buf = ''
		while True:
			try:
				data = (await loop.sock_recv(client_socket, 1024)).decode('UTF-8')
			except Exception:
				print('Disconnected')
				break
			buf, detected = await extract_json(buf + data)
			if len(detected) == 0:
				continue
			async with self.lock:
				for now in detected:
					for ind, target in enumerate(now['src']):
						# if target['id'] == 0 or target['tag'] != "dynamic" or target['h'] == -114514.0:
						if target['id'] == 0:
							continue
						if abs(target['z'] - 0.5) > self.max_z or abs(target['z'] - 0.5) < self.min_z:
							continue
						await self.update(target, ind)
				await self.clean_inactive()

	def submit(self, data):
		data = np.frombuffer(data, dtype=np.int8).reshape(-1, 2)
		for ind in range(4):
			if not isinstance(self.waves[ind], np.ndarray):
				self.waves[ind] = data[ind::4]
			else:
				self.waves[ind] = np.append(self.waves[ind], data[ind::4], axis=0)
		if self.waves[0].shape[0] * 2 >= self.hop:
			for ind in range(4):
				self.frame[ind] = self.waves[ind].flatten()[:self.hop]
				self.waves[ind] = None
			stream.write(bytes(self.frame[0]))

	async def handle_audio(self, client_socket, loop):
		while True:
			try:
				data = await loop.sock_recv(client_socket, 4096)
			except Exception:
				print('Disconnected')
				break
			async with self.frame_lock:
				await loop.run_in_executor(None, lambda: self.submit(data))

	async def get_frame_async(self, loop):
		data = self.frame
		self.frame = [None for i in range(self.max_sounds)]
		return data

	async def listen(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind((self.host, self.port))
		server.listen(1)
		server.setblocking(False)

		while True:
			client_socket, address = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('SST Client from {}'.format(address))
			loop.create_task(self.handle(client_socket, loop))

	async def listen_audio(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind((self.host, self.port_audio))
		server.listen(1)
		server.setblocking(False)

		while True:
			client_socket, address = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('Audio Client from {}'.format(address))
			loop.create_task(self.handle_audio(client_socket, loop))

class SSL:
	def __init__(self):
		self.config = config['ssl']
		self.lock = asyncio.Lock()
		self.angle = 0.0
		self.host = self.config['host']
		self.port = self.config['port']
		self.offset = self.config['offset']

	async def update(self, target):
		x, y, e = int(float(target['x']) * 400), int(-float(target['y']) * 400), int(float(target['E']) * 255)
		if (x in range(-60, 60) and y in range(-60, 60)) or e < 115:
			return
		self.angle = (360 + math.degrees(math.atan2(y, x)) + self.offset) % 360

	async def handle(self, client_socket, loop):
		buf = ''
		while True:
			try:
				data = (await loop.sock_recv(client_socket, 1024)).decode('UTF-8')
			except Exception:
				print('SSL client disconnected.')
				break
			buf, detected = await extract_json(buf + data)
			async with self.lock:
				for data in detected:
					for target in data['src']:
						await self.update(target)

	async def listen(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind((self.host, self.port))
		server.listen(1)
		server.setblocking(False)

		while True:
			(client_socket, address) = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('SSL client from {}'.format(address))
			loop.create_task(self.handle(client_socket, loop))

class WS:
	def __init__(self):
		self.config = config['ws']
		self.host = self.config['host']
		self.port = self.config['port']
		self.fps = self.config['fps']
		self.ttl = self.config['ttl']
		self.lock = asyncio.Lock()
		self.trusted_label = []

	async def doa(self, websocket):
		while True:
			async with self.sst.lock:
				pos = [self.sst.pos[i] for i in self.sst.pos]
			async with self.ssl.lock:
				angle = self.ssl.angle
			async with self.cls.lock:
				label, acc = self.cls.label, self.cls.acc
			async with self.lock:
				self.trusted_label = [i for i in self.trusted_label if time.time() - i[1] <= self.ttl]
				if len(self.trusted_label) != 0:
					label2 = self.trusted_label[-1][0]
				else:
					label2 = None

			for i in range(len(pos)):
				pos[i]['category'] = label[pos[i]['order']]
				pos[i]['acc'] = acc[pos[i]['order']]

			if label2 is not None and len(pos) != 0:
				target = 0
				for i in range(len(pos)):
					if abs(pos[i]['angle'] - angle) < 11:
						target = i
						break
				pos[target]['category'] = label2

			result = json.dumps({'sound': pos, 'angle': angle})

			try:
				await websocket.send(result)
			except Exception as e:
				print("Websocket client connection closed.")
				break
			await asyncio.sleep(1 / self.fps)

	async def odas(self, sst, ssl, loop):
		sst_task = asyncio.create_task(sst.listen(loop))
		sst_audio_task = asyncio.create_task(sst.listen_audio(loop))
		ssl_task = asyncio.create_task(ssl.listen(loop))

		await asyncio.gather(sst_task, sst_audio_task, ssl_task)

	async def category(self, websocket):
		data = json.loads(await websocket.recv())
		async with self.lock:
			self.trusted_label.append((data['category'], time.time()))

	async def handle(self, websocket, path):
		if path == '/category':
			await self.category(websocket)
		else:
			print("Client connected.")
			sst_doa_task = asyncio.create_task(self.doa(websocket))

			await asyncio.gather(sst_doa_task)

	async def announce(self):
		ip = ''
		while True:
			try:
				ip = netifaces.ifaddresses(get_connection_name_from_guid()[config['network']['interface']])[AF_INET][0]['addr']
				break
			except Exception as e:
				await asyncio.sleep(1)
				continue
		while True:
			try:
				async with aiohttp.ClientSession() as session:
					async with session.get('http://swc.otmdb.cn/announce?name=sound&ip=ws%3A%2F%2F{}%3A{}'.format(ip, self.port)) as resp:
						if resp.status != 200:
							continue
						await session.get('http://swc.otmdb.cn/announce?name=sound_server&ip={}'.format(ip))
				break
			except Exception as e:
				await asyncio.sleep(1)
				continue

	async def main(self):
		print('Starting ODAS Server...')
		loop = asyncio.get_event_loop()
		self.cls = CLS()
		self.sst = SST()
		self.ssl = SSL()

		odas_task = asyncio.create_task(self.odas(self.sst, self.ssl, loop))
		announce_task = asyncio.create_task(self.announce())
		cls_task = asyncio.create_task(self.cls.handle(self.sst, self.ssl, loop))
		ws_server = await websockets.serve(self.handle, self.host, self.port)

		await asyncio.gather(odas_task, announce_task, cls_task, ws_server.wait_closed())

if __name__ == '__main__':
	config = load_config()
	ws = WS()
	asyncio.run(ws.main())