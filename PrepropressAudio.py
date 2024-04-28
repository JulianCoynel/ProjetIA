# -*- coding: utf-8 -*-
import os
import tensorflow as tf

# Defining the squeeze function
def squeeze(audio):
    audio = tf.squeeze(audio, axis=-1)
    return audio


def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.

  audio, default_audio_rate = tf.audio.decode_wav(contents=audio_binary,desired_samples=16000)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  audio = squeeze(audio)
  return audio


#Convertir des formes d'onde en spectrogrammes
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def preprocessAudio(audios):
    audio_ds = tf.data.Dataset.from_tensor_slices(audios)
    output_ds = audio_ds.map(
        map_func=decode_audio,
        num_parallel_calls=tf.data.AUTOTUNE)
    spectrogram = output_ds.map(
        map_func=get_spectrogram,
        num_parallel_calls=tf.data.AUTOTUNE)
    return spectrogram

