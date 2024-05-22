import websocket
import json

def on_message(ws, message):
    # This function will be called when a message is received
    data = json.loads(message)
    print("Received:", [i['category'] for i in data['sound']], data['category'])

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket opened")

if __name__ == "__main__":
    # WebSocket URL
    ws_url = "ws://192.168.31.217:11451"

    # Create WebSocket connection
    ws = websocket.WebSocketApp(ws_url,
                                on_message = on_message,
                                on_error = on_error,
                                on_close = on_close)
    ws.on_open = on_open

    # Run WebSocket connection infinitely
    ws.run_forever()