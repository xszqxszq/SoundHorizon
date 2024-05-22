import aiohttp
import aiofiles
import asyncio
import netifaces
import subprocess
import RPi.GPIO as GPIO
from pixel_ring import pixel_ring
from aiohttp import web

mic_color = 0x000000
HOST = '0.0.0.0'
PORT = 16666
PIN = 14

def setup():
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(PIN, GPIO.OUT)

def setPin(mode):
	GPIO.output(PIN, mode)

routes = web.RouteTableDef()

@routes.get('/fan')
async def control_fan(request):
	status = request.query['status'].lower() != 'false'
	loop = asyncio.get_running_loop()
	status = await loop.run_in_executor(None, lambda: setPin(status))
	return web.json_response({'status': True, 'data': status})

async def get_server_ip():
	ip = ''
	while True:
		try:
			ip = netifaces.ifaddresses('wlan0')[netifaces.AF_INET][0]['addr']
			break
		except Exception:
			await asyncio.sleep(1)
			continue
	while True:
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get('http://swc.otmdb.cn/ip') as resp:
					if resp.status != 200:
						continue
					ips = (await resp.json())['data']
					if 'sound_server' not in ips:
						await asyncio.sleep(1)
						continue
			result = ips['sound_server']
			break
		except Exception as e:
			await asyncio.sleep(1)
			continue
	async with aiohttp.ClientSession() as session:
		await session.get('http://swc.otmdb.cn/announce?name=raspberry&ip={}'.format(ip))
		await session.get('http://swc.otmdb.cn/announce?name=raspberry_fan&ip=http://{}:{}'.format(ip, PORT))
	return result

async def main():
	loop = asyncio.get_running_loop()
	server = await get_server_ip()
	async with aiofiles.open('/root/server/template.cfg', mode='r') as f:
		config = await f.read()
	async with aiofiles.open('/root/config.cfg', mode='w') as f:
		await f.write(config.replace('{server_ip}', server))
	print('ODAS Config updated.')
	process = await asyncio.create_subprocess_exec(
		'/root/odaslive', '-c', '/root/config.cfg',
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE
	)
	print('ODAS Process started.')

	app = web.Application()
	app.add_routes(routes)
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, HOST, PORT)
	await site.start()

	await process.wait()
	
if __name__ == '__main__':
	pixel_ring.mono(mic_color)
	setup()
	asyncio.run(main())
