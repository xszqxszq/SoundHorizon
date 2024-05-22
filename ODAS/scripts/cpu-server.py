import aiohttp
import asyncio
import json
import math
import netifaces
import os
import psutil
import pyaudio
import socket
import struct
import subprocess
import time
import websockets
import numpy as np
from netifaces import AF_INET
from openvino.inference_engine import IECore
from pixel_ring import pixel_ring

mic_color = 0x000000
fps = 1/60

TMP = 4

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

def get_mic_index(name):
	p = pyaudio.PyAudio()
	devices = p.get_device_count()
	for i in range(devices):
		device_info = p.get_device_info_by_index(i)
		if device_info.get('maxInputChannels') > 0 and device_info.get('name') == name:
			return i


class SSL_CLS:
	def __init__(self, sss_cls):
		self.sss_cls = sss_cls
		self.lock = asyncio.Lock()
		self.mic_name = 'default'
		self.mic_index = get_mic_index(self.mic_name)	
		self.audio = pyaudio.PyAudio()
		self.chunk = np.zeros(self.sss_cls.input_shape, dtype=np.float32)
		self.record_stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sss_cls.sample_rate, input =True, frames_per_buffer=self.sss_cls.input_size, input_device_index=self.mic_index)
		self.label, self.acc = 'silence', 0.0

	def read_chunk(self):
		raw = self.record_stream.read(num_frames=self.sss_cls.hop)
		if len(raw) < 12800 * 2:
			return False

		try:
			input_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * 8.0
			scale = np.std(input_audio)
			scale = 4000 if scale < 4000 else scale
			input_audio = (input_audio - np.mean(input_audio)) / scale
			input_audio = np.reshape(input_audio, (1, 1, 1, self.sss_cls.hop))

			self.chunk[:, :, :, :self.sss_cls.overlap] = self.chunk[:, :, :, -self.sss_cls.overlap:]
			self.chunk[:, :, :, self.sss_cls.overlap:] = input_audio
		except Exception as e:
			return False
		return True

	def infer(self):
		label_txt, acc = 'silence', 0.0
		output = self.sss_cls.exec_net.infer(inputs={self.sss_cls.input_blob: self.chunk})
		output = output[self.sss_cls.output_blob]
		for batch, data in enumerate(output):
			label = np.argmax(data)
			if data[label] < 0.8:
				label_txt, acc = 'silence', 0.0
				illust_idx = 99
			else:
				label_txt, acc = self.sss_cls.labels[label], float(data[label])
				illust_idx = label
		if label_txt not in self.sss_cls.whitelist:
			label_txt = 'silence'
		return label_txt, acc

	async def infer_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.infer())

	async def read_chunk_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.read_chunk())

	async def handle(self, loop):
		last = 'silence'

		while True:
			if not (await self.read_chunk_async(loop)):
				continue
			label, acc = await self.infer_async(loop)
			async with self.lock:
				self.label, self.acc = label, acc

class SSS_CLS:
	def __init__(self):
		self.lock = asyncio.Lock()
		device = 'CPU'
		self.mic_name = 'default'
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
		self.chunk = [np.zeros(self.input_shape, dtype=np.float32) for i in range(TMP)]
		self.hop = int(self.input_size * 0.8)
		self.overlap = int(self.input_size - self.hop)
		self.label, self.acc = ['silence' for i in range(TMP)], [0.0 for i in range(TMP)]

	def update_audio(self, raw):
		for i in range(TMP):
			try:
				input_audio = np.frombuffer(raw[i], dtype=np.int16).astype(np.float32) * 8.0
				scale = np.std(input_audio)
				scale = 4000 if scale < 4000 else scale
				input_audio = (input_audio - np.mean(input_audio)) / scale
				input_audio = np.reshape(input_audio, (1, 1, 1, self.hop))

				self.chunk[i][:, :, :, :self.overlap] = self.chunk[i][:, :, :, -self.overlap:]
				self.chunk[i][:, :, :, self.overlap:] = input_audio
			except Exception as e:
				return False
		return True

	async def read_chunk(self, loop):
		while True:
			async with self.sss.frame_lock:
				if len(self.sss.frame) != 0:
					raw = await self.sss.get_frame_async(loop)
					break
			await asyncio.sleep(1/120)
		return await loop.run_in_executor(None, self.update_audio, raw)

	def infer(self):
		label_txt, acc = ['silence' for i in range(TMP)], [0.0 for i in range(TMP)]
		for i in range(TMP):
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
		for i in range(TMP):
			if label_txt[i] not in self.whitelist:
				label_txt[i] = 'silence'
		return label_txt, acc

	async def infer_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.infer())

	async def handle(self, sst, ssl, loop):
		last = ['silence' for i in range(TMP)]
		self.sss = sst

		while True:
			if not (await self.read_chunk(loop)):
				continue
			label, acc = await self.infer_async(loop)
			async with self.lock:
				self.label, self.acc = label, acc

class SSS:
	def __init__(self, port=11450, port_audio=11453):
		self.lock = asyncio.Lock()
		self.pos = {}
		self.keep_after_seconds = 0.1
		self.activity_threshold = 0.5
		self.port = port
		self.port_audio = port_audio
		self.frame_lock = asyncio.Lock()
		self.waves = [[] for i in range(4)]
		self.frame = [[] for i in range(4)]

	async def clean_inactive(self):
		self.pos = dict([(i, self.pos[i]) for i in self.pos if time.time() - self.pos[i]['last_activity'] < self.keep_after_seconds])

	async def update(self, target, ind):
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

	async def handle(self, client_socket, loop):
		buf = ''
		while True:
			try:
				data = (await loop.sock_recv(client_socket, 1024)).decode('UTF-8')
			except Exception:
				print('Disconnected')
				break
			buf, detected = await extract_json(buf + data)
			async with self.lock:
				for data in detected:
					for ind, target in enumerate(data['src']):
						if target['id'] == 0:
							continue
						await self.update(target, ind)
				await self.clean_inactive()

	def submit(self):
		for ind in range(4):
			self.frame[ind] = self.waves[ind][-12800*2:]
			self.waves[ind] = []

	async def handle_audio(self, client_socket, loop):
		while True:
			try:
				data = await loop.sock_recv(client_socket, 1024)
			except Exception:
				print('Disconnected')
				break
			for i in range(0, len(data), 8):
				for ind in range(4):
					self.waves[ind].append(data[i+ind*2])
					self.waves[ind].append(data[i+ind*2+1])
			if len(self.waves[0]) // 2 > 12800:
				async with self.frame_lock:
					await loop.run_in_executor(None, self.submit)

	def get_frame(self):
		return [bytes(i[:12800*2]) for i in self.frame]

	async def get_frame_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.get_frame())

	async def listen(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port))
		server.listen(1)
		server.setblocking(False)

		while True:
			client_socket, address = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('SSS Client from {}'.format(address))
			loop.create_task(self.handle(client_socket, loop))

	async def listen_audio(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port_audio))
		server.listen(1)
		server.setblocking(False)

		while True:
			client_socket, address = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('Audio Client from {}'.format(address))
			loop.create_task(self.handle_audio(client_socket, loop))

class SSL:
	def __init__(self, port=11452):
		self.lock = asyncio.Lock()
		self.angle = 0.0
		self.dist = 0.0
		self.port = port

	async def update(self, target):
		x, y, e = int(float(target['x']) * 400), int(-float(target['y']) * 400), int(float(target['E']) * 255)
		if (x in range(-60, 60) and y in range(-60, 60)) or e < 100:
			return
		self.angle, self.dist = math.degrees(math.atan2(y, x)) + 180, math.sqrt(x ** 2 + y ** 2)

	async def handle(self, client_socket, loop):
		buf = ''
		while True:
			try:
				data = (await loop.sock_recv(client_socket, 1024)).decode('UTF-8')
			except Exception:
				print('Disconnected')
				break
			buf, detected = await extract_json(buf + data)
			async with self.lock:
				for data in detected:
					for target in data['src']:
						await self.update(target)

	async def listen(self, loop):
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		server.bind(('0.0.0.0', self.port))
		server.listen(1)
		server.setblocking(False)

		while True:
			(client_socket, address) = await loop.sock_accept(server)
			optval = struct.pack("ii", 1, 0)
			client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, optval)
			print('SSL Client from {}'.format(address))
			loop.create_task(self.handle(client_socket, loop))

class WS:
	async def doa(self, websocket):
		while True:
			try:
				async with self.sst.lock:
					pos = [self.sst.pos[i] for i in self.sst.pos]
				async with self.ssl.lock:
					angle, dist = self.ssl.angle, self.ssl.dist
				async with self.classify.lock:
					label, acc = self.classify.label, self.classify.acc
				async with self.classify2.lock:
					label2, acc2 = self.classify2.label, self.classify2.acc
				for i in range(len(pos)):
					pos[i]['category'] = label[pos[i]['order']]
					pos[i]['acc'] = acc[pos[i]['order']]
				result = json.dumps({'sound': pos, 'angle': angle, 'dist': dist, 'category': label2})
				await websocket.send(result)
			except Exception:
				print("Client connection closed.")
				break
			await asyncio.sleep(fps)

	async def odas(self, sst, ssl, loop):
		try:
			sst_task = asyncio.create_task(sst.listen(loop))
			sst_audio_task = asyncio.create_task(sst.listen_audio(loop))
			ssl_task = asyncio.create_task(ssl.listen(loop))

			process = await asyncio.create_subprocess_exec(
				'/root/odaslive', '-c', '/root/config.cfg',
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE
			)

			await asyncio.gather(sst_task, sst_audio_task, ssl_task, process.wait())
		except asyncio.CancelledError:
			process.terminate()

	async def handle(self, websocket, path):
		print("Client connected.")
		sst_doa_task = asyncio.create_task(self.doa(websocket))

		await asyncio.gather(sst_doa_task)

	async def announce(self):
		ip = ''
		while True:
			try:
				ip = netifaces.ifaddresses('wlan0')[AF_INET][0]['addr']
				break
			except Exception:
				await asyncio.sleep(1)
				continue
		while True:
			try:
				async with aiohttp.ClientSession() as session:
					async with session.get('http://swc.otmdb.cn/announce?name=sound&ip=ws%3A%2F%2F{}%3A{}'.format(ip, self.port)) as resp:
						if resp.status != 200:
							continue
				break
			except Exception as e:
				await asyncio.sleep(1)
				continue

	async def main(self):
		loop = asyncio.get_event_loop()
		self.classify = SSS_CLS()
		self.classify2 = SSL_CLS(self.classify)
		self.sst = SSS()
		self.ssl = SSL()
		self.port = 11451

		odas_task = asyncio.create_task(self.odas(self.sst, self.ssl, loop))
		announce_task = asyncio.create_task(self.announce())
		cls_task = asyncio.create_task(self.classify.handle(self.sst, self.ssl, loop))
		cls2_task = asyncio.create_task(self.classify2.handle(loop))
		ws_server = await websockets.serve(self.handle, '0.0.0.0', self.port)

		await asyncio.gather(odas_task, announce_task, cls_task, cls2_task, ws_server.wait_closed())

pixel_ring.mono(mic_color)

ws = WS()
print('Starting Server...')
asyncio.run(ws.main())