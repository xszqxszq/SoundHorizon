import pyaudio
import webrtcvad

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 10  # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read from microphone
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 3
NUM_WINDOW_CHUNKS_START = NUM_WINDOW_CHUNKS * 3
SAMPLE_WIDTH = 2

vad = webrtcvad.Vad(1)

def read_chunks(stream):
    while True:
        chunk = stream.read(CHUNK_SIZE)
        yield chunk

def is_speech(chunk):
    return vad.is_speech(chunk, RATE)

def main():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    print("* Listening for voice activity...")

    num_voiced = 0
    num_unvoiced = 0

    for i, chunk in enumerate(read_chunks(stream)):
        is_speech_chunk = is_speech(chunk)

        if is_speech_chunk:
            num_voiced += 1
            num_unvoiced = 0
        else:
            num_unvoiced += 1
            num_voiced = 0

        if num_voiced > NUM_WINDOW_CHUNKS_END:
            print("* Voice detected!")
            num_voiced = 0
        elif num_unvoiced > NUM_WINDOW_CHUNKS_START:
            print("* Silence detected!")
            num_unvoiced = 0

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()
