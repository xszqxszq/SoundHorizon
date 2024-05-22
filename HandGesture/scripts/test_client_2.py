import asyncio
import websockets
import json
import cv2

cap = cv2.VideoCapture(2)
async def send_image(websocket):
    while True:
        success, image = cap.read()
        _, data = cv2.imencode('.jpg', image)
        data = data.tobytes()

        # Send the length of the image data as a 4-byte unsigned integer
        await websocket.send(len(data).to_bytes(4, byteorder='big'))

        # Send the image data
        await websocket.send(data)

        await asyncio.sleep(1/60)  # Adjust the interval as needed

async def receive_json(websocket):
    while True:
        # Receive JSON data from the server
        json_data = await websocket.recv()

        # Process the received JSON data
        print("Received JSON:", json_data)

async def main():
    async with websockets.connect('ws://localhost:11454') as websocket:
        send_task = asyncio.create_task(send_image(websocket))
        receive_task = asyncio.create_task(receive_json(websocket))
        await asyncio.gather(send_task, receive_task)

asyncio.run(main())