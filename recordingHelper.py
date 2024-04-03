# -*- coding: utf-8 -*-
import pyaudio
import numpy as np
from tensorflow import constant

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 16000
p = pyaudio.PyAudio()

def recordAudio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    # print("recording stopped")

    stream.stop_stream()
    stream.close()
    r = constant(np.frombuffer(b''.join(frames), dtype=np.float32))
    return r


def terminate():
    p.terminate()