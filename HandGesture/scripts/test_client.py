import asyncio
import websockets
import json
import cv2

async def receive_json(websocket):
    while True:
        # Receive JSON data from the server
        json_data = await websocket.recv()

        # Process the received JSON data
        print("Received JSON:", json_data)

async def main():
    async with websockets.connect('ws://localhost:11454') as websocket:
        receive_task = asyncio.create_task(receive_json(websocket))
        await asyncio.gather(receive_task)

asyncio.run(main())