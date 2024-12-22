import librosa
import numpy as np
import streamlit as st

from glob import glob
import tensorflow as tf
import IPython

def prepare_audio_for_chord_detection(audio_path = "test.wav", segment_duration=2.0, hop_length=128, sr = 10000):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Define segment length in samples
    segment_length = int(segment_duration * sr)
    
    # Prepare list to store CQT chroma for each segment
    segments = []
    
    # Slide a window through the audio
    for start in range(0, len(y) - segment_length, segment_length):
        # Extract segment
        segment = y[start:start + segment_length]
        
        # Compute CQT chroma for the segment
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, hop_length=hop_length)
        
        segments.append({
            'start_time': start / sr,
            'end_time': (start + segment_length) / sr,
            'chroma': chroma
        })
    
    return segments



def store_chords(features, loaded_model):
    chordlist = []
    for i in range(0, len(features)):
        feature = features[i]["chroma"]
        feature_T = feature.transpose(1,0)
        feature_T = feature_T.reshape(1, feature_T.shape[0], feature_T.shape[1])
        y_pred = loaded_model.predict(feature_T)
        chord_mapping = {
            0: 'Am', 
            1: 'Bb', 
            2: 'Bdim', 
            3: 'C',
            4: 'Dm',
            5: 'Em',
            6: 'F',
            7: 'G'
        }
        if len(chordlist) == 0 or chordlist[-1] != chord_mapping.get(np.argmax(y_pred), 'Unknown') :
            chordlist.append(chord_mapping.get(np.argmax(y_pred), 'Unknown'))
    return chordlist

def print_all_chords(chordlist):
   chordStr = ' '.join([str(item) for item in chordlist])
   st.text(chordStr)


def load_model(path_to_model = 'chord_detection_model.h5'):
   loaded_model = tf.keras.models.load_model( path_to_model, compile=True)
   return loaded_model
