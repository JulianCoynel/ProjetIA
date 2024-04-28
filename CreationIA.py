# -*- coding: utf-8 -*-
import argparse
import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import pathlib
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import PrepropressAudio as pa



# Defining the squeeze function
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels



# Convert waveform to spectrogram
def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	return spectrogram[..., tf.newaxis]

#Creating spectrogram dataset from waveform or audio data
def get_spectrogram_dataset(dataset):
	dataset = dataset.map(
		lambda x, y: (pa.get_spectrogram(x), y),
		num_parallel_calls=tf.data.AUTOTUNE)
	return dataset



# RNN
def get_model(input_shape, num_labels):
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=input_shape),
		tf.keras.layers.Normalization(),

		# 2 LSTM layers
		tf.keras.layers.Reshape((124, 129), input_shape=input_shape),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

		# Dense layer
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dropout(0.5),

		# Softmax layer to get the label prediction
		tf.keras.layers.Dense(num_labels, activation='softmax')
	])
	# Printing model summary
	model.summary()
	return model



def run(dataset_path,model_path):
    DATASET_PATH = dataset_path #'../Données/MotsSimple'

    # Using audio_dataset_from_directory function to create dataset with audio data
    training_set, validation_set = tf.keras.utils.audio_dataset_from_directory(
    	directory=DATASET_PATH,
    	#batch_size=512,
    	validation_split=0.2,
    	output_sequence_length=16000,
    	seed=0,
    	subset='both')

    # Extracting audio labels
    label_names = np.array(training_set.class_names)
    
    
    # Applying the function on the dataset obtained from previous step
    training_set = training_set.map(lambda x, y: (pa.squeeze(x), y), tf.data.AUTOTUNE)
    validation_set = validation_set.map(lambda x, y: (pa.squeeze(x), y), tf.data.AUTOTUNE)
    
    # Applying the function on the audio dataset
    train_set = get_spectrogram_dataset(training_set)
    validation_set = get_spectrogram_dataset(validation_set)

    # Dividing validation set into two equal val and test set
    val_set = validation_set.take(validation_set.cardinality() // 2)
    test_set = validation_set.skip(validation_set.cardinality() // 2)

    train_set_shape = train_set.element_spec[0].shape
    val_set_shape = val_set.element_spec[0].shape
    test_set_shape = test_set.element_spec[0].shape
    
    # Getting input shape from the sample audio and number of classes
    input_shape = next(iter(train_set))[0][0].shape
    print("Input shape:", input_shape)
    num_labels = len(label_names)

    # Creating a model
    model = get_model(input_shape, num_labels)

    model.compile(
    	optimizer=tf.keras.optimizers.Adam(),
    	loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    	metrics=['accuracy'],
    )

    EPOCHS = 20
    history = model.fit(
    	train_set,
    	validation_data=val_set,
    	epochs=EPOCHS,
    )

    # Plotting the history
    metrics = history.history
    plt.figure(figsize=(10, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Confusion matrix
    y_pred = np.argmax(model.predict(test_set), axis=1)
    y_true = np.concatenate([y for x, y in test_set], axis=0)
    cm = tf.math.confusion_matrix(y_true, y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    report = classification_report(y_true, y_pred)
    print(report)
    
    try:  
        os.mkdir(model_path)  
    except OSError as error:
        print(error)
    
    listLabelF = open(model_path+'/label.txt','w')
    for l in label_names :
        listLabelF.write(l+'\n')
    listLabelF.close()

    model.save(model_path+'/model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AI with the given dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path where the dataset is")
    parser.add_argument("--save_path", type=str, help="Path where the model will be saved")

    args = parser.parse_args()

    if args.dataset_path is None or args.save_path is None:
        print("Please provide both --dataset_path and --save_path")
    else:
        
        run(args.dataset_path,args.save_path)