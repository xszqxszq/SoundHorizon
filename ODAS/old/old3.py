import aiohttp
import asyncio
import json
import math
import netifaces
import os
import psutil
import pyaudio
import socket
import subprocess
import threading
import time
import usb.core
import usb.util
import websockets
import numpy as np
from netifaces import AF_INET
from openvino.inference_engine import IECore
from pixel_ring import pixel_ring
from tuning import Tuning

mic_color = 0x000000
fps = 1/20

def is_process_running(name):
	return any([i.info['name'] == name for i in psutil.process_iter(['pid', 'name', 'status'])])

def extract_json(buf):
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
	iface_guids = ni.interfaces()
	iface_names = dict([(iface_guids[i], '(unknown)') for i in range(len(iface_guids))])
	reg = wr.ConnectRegistry(None, wr.HKEY_LOCAL_MACHINE)
	reg_key = wr.OpenKey(reg, r'SYSTEM\CurrentControlSet\Control\Network\{4d36e972-e325-11ce-bfc1-08002be10318}')
	for iface in iface_guids:
		try:
			reg_subkey = wr.OpenKey(reg_key, iface + r'\Connection')
			iface_names[iface] = wr.QueryValueEx(reg_subkey, 'Name')[0]
		except FileNotFoundError:
			pass
	return {v: k for k, v in iface_names.items()}

class SSS_CLS:
	def __init__(self):
		self.lock = threading.Lock()
		device = 'CPU'
		mic_name = 'default'
		model_path = "public/aclnet-int8/aclnet_des_53_int8.xml"
		label_path = "data/aclnet_53cl.txt"
		self.whitelist = open('data/whitelist.txt', 'r').read().strip().split('\n')
		ie = IECore()
		net = ie.read_network(model_path, model_path[:-4] + ".bin")
		self.input_blob = next(iter(net.input_info))
		self.input_shape = net.input_info[self.input_blob].input_data.shape
		self.output_blob = next(iter(net.outputs))
		self.exec_net = ie.load_network(network=net, device_name=device)
		self.labels = []
		with open(label_path, "r") as file:
			self.labels = [line.rstrip() for line in file.readlines()]
		self.input_size = self.input_shape[-1]
		self.sample_rate = 16000
		self.chunk = [np.zeros(self.input_shape, dtype=np.float32) for i in range(4)]
		self.hop = int(self.input_size * 0.8)
		self.overlap = int(self.input_size - self.hop)
		self.label, self.acc = ['silence' for i in range(4)], [0.0 for i in range(4)]

	def read_chunk(self):
		while True:
			with self.sss.frame_lock:
				if len(self.sss.frame) != 0:
					raw = [bytes(i[:12800*2]) for i in self.sss.frame]
					break
			time.sleep(1/30)

		for i in range(4):
			try:
				input_audio = np.frombuffer(raw[i], dtype=np.int16).astype(np.float32) * 8.0
				scale = np.std(input_audio)
				scale = 4000 if scale < 4000 else scale
				input_audio = (input_audio - np.mean(input_audio)) / scale
				input_audio = np.reshape(input_audio, (1, 1, 1, self.hop))

				self.chunk[i][:, :, :, :self.overlap] = self.chunk[i][:, :, :, -self.overlap:]
				self.chunk[i][:, :, :, self.overlap:] = input_audio
			except Exception:
				return False
		return True

	def infer(self):
		label_txt, acc = ['silence' for i in range(4)], [0.0 for i in range(4)]
		for i in range(4):
			output = self.exec_net.infer(inputs={self.input_blob: self.chunk[i]})
			output = output[self.output_blob]
			for batch, data in enumerate(output):
				label = np.argmax(data)
				if data[label] < 0.8:
					label_txt[i], acc[i] = 'silence', 0.0
					illust_idx = 99
				else:
					label_txt[i], acc[i] = self.labels[label], float(data[label])
					illust_idx = label
		for i in range(4):
			if label_txt[i] not in self.whitelist:
				label_txt[i] = 'silence'
		return label_txt, acc

	def handle(self, websocket, sst, ssl):
		last = ['silence' for i in range(4)]
		self.sss = sst

		while True:
			if not self.read_chunk():
				continue
			label, acc = self.infer()
			with self.lock:
				self.label, self.acc = label, acc
			if any([label[i] != last[i] for i in range(4)]):
				last = label
				result = ''
				with sst.lock:
					pos = [sst.pos[i] for i in sst.pos]
				with ssl.lock:
					angle, dist = ssl.angle, ssl.dist
				for i in range(len(pos)):
					pos[i]['category'] = label[pos[i]['order']]
					pos[i]['acc'] = acc[pos[i]['order']]
				result = {
					'sound': pos,
					'angle': angle,
					'dist': dist
				}
				try:
					asyncio.run(websocket.send(json.dumps(result)))
				except Exception:
					print("Client connection closed.")
					break

class SSS:
	def __init__(self, port=11450, port_audio=11453):
		self.lock = threading.Lock()
		self.pos = {}
		self.keep_after_seconds = 0.5
		self.activity_threshold = 0.1
		self.port = port
		self.port_audio = port_audio
		self.wave_lock = threading.Lock()
		self.frame_lock = threading.Lock()
		self.waves = [[] for i in range(4)]
		self.now_frames = 0
		self.frame = [[] for i in range(4)]

	def clean_inactive(self):
		self.pos = dict([(i, self.pos[i]) for i in self.pos if time.time() - self.pos[i]['last_activity'] < self.keep_after_seconds])

	def update(self, target, ind):
		x, y, z = int(float(target['x']) * 400), int(-float(target['y']) * 400), int(float(target['z'] - 0.5) * 400)
		if target['id'] in self.pos and target['activity'] < self.activity_threshold:
			last_activity = self.pos[target['id']]['last_activity']
		else:
			last_activity = time.time()
		angle, dist = math.degrees(math.atan2(y, x)) + 180, math.sqrt(x ** 2 + y ** 2)
		self.pos[target['id']] = {
			'id': target['id'], 
			'x': x + 400, 'y': y + 400, 
			'z': z, 'activity': target['activity'],
			'last_activity': last_activity, 
			'angle': angle, 
			'dist': dist,
			'order': ind
		}

	def handle(self, client_socket):
		buf = ''
		while True:
			try:
				data = client_socket.recv(1024).decode('UTF-8')
			except Exception:
				print('Disconnected')
				break
			buf, detected = extract_json(buf + data)
			with self.lock:
				for data in detected:
					for ind, target in enumerate(data['src']):
						if target['id'] == 0:
							continue
						self.update(target, ind)
				self.clean_inactive()

	def handle_audio(self, client_socket):
		while True:
			try:
				data = client_socket.recv(1024)
				if len(data) < 1024:
					continue
			except Exception:
				print('Disconnected')
				break
			with self.wave_lock:
				for i in range(0, 1024, 8):
					for ind in range(4):
						self.waves[ind].append(data[i+ind*2])
						self.waves[ind].append(data[i+ind*2+1])
				self.now_frames = len(self.waves[0]) // 2
				if self.now_frames > 12800:
					with self.frame_lock:
						for ind in range(4):
							self.frame[ind] = self.waves[ind][-12800*2:]
							self.waves[ind] = []
						self.now_frames = 0

	def listen(self):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port))
		server.listen(1)
		while True:
			(client_socket, address) = server.accept()
			print('SST Client from {}'.format(address))
			threading.Thread(target=self.handle, args=(client_socket,)).start()

	def listen_audio(self):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port_audio))
		server.listen(1)
		while True:
			(client_socket, address) = server.accept()
			print('SSS Client from {}'.format(address))
			threading.Thread(target=self.handle_audio, args=(client_socket,)).start()

class SSL:
	def __init__(self, port=11452):
		self.lock = threading.Lock()
		self.angle = 0.0
		self.dist = 0.0
		self.port = port

	def update(self, target):
		x, y, e = int(float(target['x']) * 400), int(-float(target['y']) * 400), int(float(target['E']) * 255)
		if (x in range(-60, 60) and y in range(-60, 60)) or e < 100:
			return
		self.angle, self.dist = math.degrees(math.atan2(y, x)) + 180, math.sqrt(x ** 2 + y ** 2)

	def handle(self, client_socket):
		buf = ''
		while True:
			try:
				data = client_socket.recv(1024).decode('UTF-8')
			except Exception:
				print('Disconnected')
				break
			buf, detected = extract_json(buf + data)
			with self.lock:
				for data in detected:
					for target in data['src']:
						self.update(target)

	def listen(self):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port))
		server.listen(1)
		while True:
			(client_socket, address) = server.accept()
			print('SSL Client from {}'.format(address))
			threading.Thread(target=self.handle, args=(client_socket,)).start()

class WS:
	def __init__(self):
		self.classify = SSS_CLS()
		self.sst = SSS()
		self.ssl = SSL()
		self.port = 11451

	async def doa(self, websocket):
		while True:
			try:
				with self.sst.lock:
					pos = [self.sst.pos[i] for i in self.sst.pos]
				with self.ssl.lock:
					angle, dist = self.ssl.angle, self.ssl.dist
				with self.classify.lock:
					label, acc = self.classify.label, self.classify.acc
				for i in range(len(pos)):
					pos[i]['category'] = label[pos[i]['order']]
					pos[i]['acc'] = acc[pos[i]['order']]
				result = json.dumps({'sound': pos, 'angle': angle, 'dist': dist})
				await websocket.send(result)
			except Exception:
				print("Client connection closed.")
				break
			await asyncio.sleep(fps)

	async def odas(self, sst, ssl):
		try:
			sst_task = asyncio.to_thread(sst.listen)
			sst_audio_task = asyncio.to_thread(sst.listen_audio)
			ssl_task = asyncio.to_thread(ssl.listen)

			process = await asyncio.create_subprocess_exec(
				'/root/odaslive', '-c', '/root/config.cfg',
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE
			)

			await asyncio.gather(process.wait(), sst_task, sst_audio_task, ssl_task)
		except asyncio.CancelledError:
			process.terminate()

	async def handle(self, websocket, path):
		print("Client connected.")
		sst_doa_task = asyncio.create_task(self.doa(websocket))
		cls_task = asyncio.get_event_loop().run_in_executor(None, lambda: self.classify.handle(websocket, self.sst, self.ssl))

		await asyncio.gather(sst_doa_task, cls_task)

	async def announce(self):
		ip = ''
		while True:
			try:
				ip = netifaces.ifaddresses('wlan0')[AF_INET][0]['addr']
				break
			except Exception:
				asyncio.sleep(1)
				continue
		while True:
			try:
				async with aiohttp.ClientSession() as session:
					async with session.get('http://swc.otmdb.cn/announce?name=sound&ip=ws%3A%2F%2F{}%3A{}'.format(ip, self.port)) as resp:
						if resp.status != 200:
							continue
				break
			except Exception as e:
				asyncio.sleep(1)
				continue

	async def main(self):
		odas_task = asyncio.create_task(self.odas(self.sst, self.ssl))
		announce_task = asyncio.create_task(self.announce())
		ws_server = await websockets.serve(self.handle, '0.0.0.0', self.port)

		await asyncio.gather(odas_task, announce_task, ws_server.wait_closed())

# if not is_process_running('pulseaudio'):
# 	os.system('pulseaudio -D')
pixel_ring.mono(mic_color)

ws = WS()
print('Starting Server...')
asyncio.run(ws.main())