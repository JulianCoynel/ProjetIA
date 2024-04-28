# -*- coding: utf-8 -*-
import argparse
from datasets import load_dataset
import numpy as np
import wave
import os

UA = load_dataset("ngdiana/uaspeech_severity_low")
UA_df =  UA['train'].to_pandas()
UA_df['filename'] = UA_df['path'].apply(lambda x: x.split("/")[-1])

def array2WAV(id):
    
    # Define the sample rate and number of samples
    sample_rate = 16000
    num_samples = 1

    # Create a directory if it doesn't exist
    target_folder = UA_df['target'].iloc[id]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    

     # Define the WAV file path
    wav_path = os.path.join(target_folder, f"{UA_df['filename'].iloc[id]}")
    
    # Create a WAV file object
    wav_file = wave.open(wav_path, "w")

    
    # Set the WAV file parameters
    wav_file.setnchannels(1) # 1 channel (mono)
    wav_file.setsampwidth(2) # 16-bit sample width
    wav_file.setframerate(sample_rate)
    
    # Write the samples to the WAV file as binary data
    
    samples = UA_df['speech'].iloc[id]
    samples = (samples * (2**15 - 1)).astype(np.int16)
    wav_file.writeframes(samples.tobytes())
    
    # Close the WAV file
    wav_file.close()
    

    
    
def run(out_dir):

    os.chdir(out_dir)

    for id in range(0, len(UA_df)):
        array2WAV(id)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Hugging face dataset.")
    parser.add_argument("--save_path", type=str, help="Path where the dataset will be saved")

    args = parser.parse_args()

    if args.save_path is None:
        print("Please provide --save_path")
    else:
        run(args.save_path)