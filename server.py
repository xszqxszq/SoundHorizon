from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import json
import uvicorn
import serial
import threading
import sys
import glob
import netifaces as ni
import winreg as wr
import aiohttp
import aiofiles
import asyncio
import yaml
import time
import websockets
from micloud import MiCloud
from miio import DeviceFactory
from netifaces import AF_INET
from typing import Optional

def serial_ports():
	if sys.platform.startswith('win'):
		ports = ['COM%s' % (i + 1) for i in range(256)]
	elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
		ports = glob.glob('/dev/tty[A-Za-z]*')
	elif sys.platform.startswith('darwin'):
		ports = glob.glob('/dev/tty.*')
	else:
		raise EnvironmentError('Unsupported platform')

	result = []
	for port in ports:
		try:
			s = serial.Serial(port)
			s.close()
			result.append(port)
		except (OSError, serial.SerialException):
			pass
	return [i for i in result if i not in ['COM3', 'COM33']]

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

def load_config():
	with open('config.yml', 'r', encoding='UTF-8') as f:
		return yaml.safe_load(f)

def init_mi_cloud():
	mc = MiCloud(config['mi_cloud']['username'], config['mi_cloud']['password'])
	mc.login()
	token = mc.get_token()
	devices = mc.get_devices(country='cn')
	try:
		bulb = next(d for d in devices if d['name'] == config['mi_cloud']['bulb_name'])
		return DeviceFactory.create(bulb['localip'], bulb['token'])
	except Exception as e:
		print(e)
	return None

def init_arduino():
	ports = serial_ports()
	print(ports)
	if len(ports) == 0:
		return None
	else:
		return serial.Serial(ports[0], 19200)


config = load_config()

HOST = config['host']
PORT = config['port']
amap_key = config['amap_key']
app = FastAPI()
arduino = init_arduino()
light_bulb = init_mi_cloud()
ips = {}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
colors_id = 0

def is_light_on():
	return light_bulb.status().__getattr__('is_on')

@app.get('/light_status')
async def control_light():
	loop = asyncio.get_running_loop()
	status = await loop.run_in_executor(None, lambda: is_light_on())
	return JSONResponse(content={'status': True, 'data': status})

def toggle_bulb():
	global colors_id
	light_bulb.actions()['Yeelight.toggle'].method()
	if is_light_on():
		light_bulb.set_rgb(colors[colors_id])
		colors_id = (colors_id + 1) % len(colors)

async def stay_active():
	while True:
		loop = asyncio.get_running_loop()
		await loop.run_in_executor(None, lambda: is_light_on())
		await asyncio.sleep(10.0)

@app.get('/light')
async def control_light(id: Optional[int] = 1):
	if id == 2:
		if arduino:
			try:
				arduino.write('111'.encode('UTF-8'))
			except Exception as e:
				print(e)
				return JSONResponse(content={'status': False, 'error': str(e)})
	elif id == 1:
		loop = asyncio.get_running_loop()
		await loop.run_in_executor(None, lambda: toggle_bulb())
	return JSONResponse(content={'status': True})

@app.get('/announce')
async def announce(name: Optional[str] = None, ip: Optional[str] = None):
	if name is None or ip is None:
		return JSONResponse(content={'status': False})
	ips[name] = ip
	return JSONResponse(content={'status': True})

@app.get('/ip')
async def list_ip():
	return JSONResponse(content={'status': True, 'data': ips})

async def getGeoCode():
	async with aiohttp.ClientSession() as session:
		async with session.get('https://restapi.amap.com/v3/ip?key={}'.format(amap_key)) as res:
			data = await res.json()
			return data['adcode']

async def getWeather():
	async with aiohttp.ClientSession() as session:
		async with session.get('https://restapi.amap.com/v3/weather/weatherInfo?key={}&city={}'.format(amap_key, await getGeoCode())) as res:
			data = await res.json()
			return data['lives'][0]

@app.get('/weather')
async def weather():
	return JSONResponse(content={'status': True, 'data': await getWeather()})

async def get_sound_name(name):
	table = {
		'horn': 'Car horn',
		'bird': 'Chirping birds',
		'cat': 'Cat'
	}
	return table[name]

@app.get('/sound')
async def sound(type: str):
	async with websockets.connect(ips['sound'] + '/category') as websocket:
		await websocket.send(json.dumps({'category': await get_sound_name(type)}))
	return JSONResponse(content={'status': True})

@app.get('/demo')
async def demo():
	async with aiofiles.open('audio.html', mode='r', encoding='UTF-8') as f:
		data = await f.read()
	return Response(content=data, media_type="text/html")

def init():
	names = get_connection_name_from_guid()
	# try:
	# 	wlan = names[[i for i in names if i.startswith('以太网')][0]]
	# 	wlan_ip = ni.ifaddresses(wlan)[AF_INET][0]['addr']
	# except Exception:
	wlan = get_connection_name_from_guid()['WLAN']
	wlan_ip = ni.ifaddresses(wlan)[AF_INET][0]['addr']
	ips['home'] = 'http://{}:{}'.format(wlan_ip, PORT)
	ips['hand_gesture'] = 'ws://{}:11454'.format(wlan_ip)

async def main():
	loop = asyncio.get_running_loop()
	uvicorn_task = loop.run_in_executor(None, lambda: uvicorn.run(app, host=HOST, port=PORT))
	stay_task = asyncio.create_task(stay_active())
	await asyncio.gather(uvicorn_task, stay_task)

if __name__ == "__main__":
	init()
	asyncio.run(main())