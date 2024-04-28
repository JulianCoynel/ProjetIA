# -*- coding: utf-8 -*-
import argparse
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import speech_recognition as sr
import PrepropressAudio as pa


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source,phrase_time_limit=2)
        a = audio.get_wav_data(convert_rate=16000)
        a = tf.constant(a)
        said = ""
        try:
            said = r.recognize_google(audio)
            print("Module prediction: "+said)
        except Exception as e:
            print("Module prediction: notfound")
    return a,said


def predict_mic(loaded_model,commands):
    audio,predict= get_audio()
    spec = pa.preprocessAudio([audio])
    for spectrogram in spec.batch(1):
        prediction = loaded_model(spectrogram)
        label_pred = np.argmax(prediction, axis=1)
        command = commands[label_pred[0]]
    return command,predict



def run(model):
    listLabelF = open(model+'/label.txt','r')
    commands = np.empty(1)
    for l in listLabelF:
        line = l.strip("\n")
        commands = np.append([commands],[line])
    
    commands = np.delete(commands, 0)
    
    print('Commands:', commands)

    
    loaded_model = models.load_model(model+'/model.h5')
    #loaded_model.summary()

    #command = predict_mic()
     
    while True:
        command,predict = predict_mic(loaded_model,commands)
        print("Our prediction:", command)
        if command == "off" or predict =="off":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a given model.")
    parser.add_argument("--model_path", type=str, help="Path where the model is")

    args = parser.parse_args()

    if args.model_path is None:
        print("Please provide --model_path")
    else:
        run(args.model_path)