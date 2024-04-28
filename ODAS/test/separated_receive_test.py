import socket
import wave

# Define host and port
HOST = '0.0.0.0'  # Standard loopback interface address (localhost)
PORT = 10000        # Port to listen on (non-privileged ports are > 1023)

def handle_client(client_socket):
    # Handle incoming data from client
    waves = [wave.open("{}.wav".format(i), "wb") for i in range(4)]
    for out in waves:
        out.setnchannels(1)
        out.setsampwidth(2) # number of bytes
        out.setframerate(16000)
    while True:
        try:
            # Receive data from the client
            data = client_socket.recv(1024)
            if not data:
                # If no data is received, client has closed the connection
                print("Client disconnected")
                break
            if len(data) < 1024:
                continue
            for i in range(0, 1024, 8):
                for ind in range(4):
                    waves[ind].writeframesraw(bytes([data[i+ind*2], data[i+ind*2+1]]))
        except Exception as e:
            print(e)
            break
        # if len(data) != 1024:
        #     continue
        # print(data)
        #print("Received:", len(data))
        # Echo back the received data
        #client_socket.sendall(data)

if __name__ == "__main__":
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((HOST, PORT))
        # Listen for incoming connections
        server_socket.listen(1)
        print("Server listening on {}:{}".format(HOST, PORT))
        # Accept incoming connections
        while True:
            # Wait for a connection
            client_socket, client_address = server_socket.accept()
            print("Connected to client:", client_address)
            # Handle the client in a separate thread
            handle_client(client_socket)
