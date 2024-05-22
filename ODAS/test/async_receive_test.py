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
import threading
from netifaces import AF_INET
from openvino.inference_engine import IECore

p = pyaudio.PyAudio()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				output=True,
				frames_per_buffer=CHUNK)

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

class SST:
	def __init__(self):
		self.config = config['sst']
		self.host = self.config['host']
		self.port = self.config['port']
		self.max_sounds = self.config['max_sounds']
		self.offset = self.config['offset']
		self.min_z, self.max_z = self.config['min_z'], self.config['max_z']
		self.port_audio = self.config['audio']['port']
		self.hop = 15600
		self.ttl = self.config['activity']['ttl']
		self.activity_threshold = self.config['activity']['threshold']

		self.lock = asyncio.Lock()
		self.pos = {}
		self.waves = [[] for i in range(self.max_sounds)]

	def submit(self, data):
		for ind in range(4):
			self.waves[ind] = data[ind::4]
		stream.write(bytes(self.waves[0]))

	async def handle_audio(self, client_socket, loop):
		while True:
			try:
				data = await loop.sock_recv(client_socket, 15600)
			except Exception:
				print('Disconnected')
				break
			data = np.frombuffer(data, dtype=np.int8).reshape(-1, 2)
			async with self.lock:
				await loop.run_in_executor(None, lambda: self.submit(data))

	def get_frame(self):
		return [bytes(i) for i in self.frame]

	async def get_frame_async(self, loop):
		return await loop.run_in_executor(None, lambda: self.get_frame())

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

class WS:
	def __init__(self):
		self.config = config['ws']
		self.host = self.config['host']
		self.port = self.config['port']
		self.fps = self.config['fps']
		self.ttl = self.config['ttl']
		self.lock = asyncio.Lock()
		self.trusted_label = []

	async def odas(self, sst, loop):
		await sst.listen_audio(loop)

	async def main(self):
		print('Starting ODAS Server...')
		loop = asyncio.get_event_loop()
		self.sst = SST()

		await self.odas(self.sst, loop)

if __name__ == '__main__':
	config = load_config()
	ws = WS()
	asyncio.run(ws.main())