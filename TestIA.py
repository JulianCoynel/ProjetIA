# -*- coding: utf-8 -*-

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import constant
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import speech_recognition as sr
import gtts as gTTS

#Jeu de données

#Dataset Julian
DATASET_PATH = '../Données/MotsSimple'
#Dataset Sacha
#DATASET_PATH = '../Données/MotsSimple'



def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE

data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
print('Commands:', commands)


#Convertir des formes d'onde en spectrogrammes
def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram




def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id



def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

def preprocessAudio(audios):
    audio_ds = tf.data.Dataset.from_tensor_slices(audios)
    output_ds = audio_ds.map(
        map_func=decode_audio,
        num_parallel_calls=AUTOTUNE)
    spectrogram = output_ds.map(
        map_func=get_spectrogram,
        num_parallel_calls=AUTOTUNE)
    return spectrogram


from pydub import AudioSegment
from pydub.playback import play

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        a = audio.get_wav_data()
        #print(r)
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))
    return a

def predict_mic():
    audio= get_audio()
    
    spec = preprocessAudio([audio])
    for spectrogram in spec.batch(1):
        prediction = loaded_model(spectrogram)
        prediction = tf.nn.softmax(prediction)
        label_pred = np.argmax(prediction, axis=1)
        command = commands[label_pred[0]]
    return command

loaded_model = models.load_model('../Model/model2.h5')
loaded_model.summary()

#command = predict_mic()
 
while True:
    command = predict_mic()
    print("Predicted label:", command)
    if command == "off":
        break
