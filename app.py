import librosa
import numpy as np
import streamlit as st

from glob import glob
import tensorflow as tf
import IPython

def extract_cqt_segment(segment, target_frames = 30, sr = 10000, hop_length = 128):
  
  cqt_features = librosa.feature.chroma_cqt(y = segment, sr = sr, n_chroma = 12, n_octaves = 7, hop_length=hop_length)
  cqt_features_input = np.zeros((12, target_frames))

  increment = cqt_features.shape[1] // target_frames

  for i in range(12):
    temp = cqt_features[i]
    index = 0
    for j in range(target_frames):
      cqt_features_input[i][j] = temp[index]
      index += increment


  return cqt_features_input


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
        # chroma = extract_cqt_segment(segment)
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
        chordlist.append(chord_mapping.get(np.argmax(y_pred), 'Unknown'))
    return chordlist

def print_all_chords(chordlist):
   for i in chordlist:
       st.text(i)


def load_model():
   loaded_model = tf.keras.models.load_model('chord_detection_model.h5', compile=True)
   return loaded_model
    


# def main():
#     # features = prepare_audio_for_chord_detection(audio_path = "wav_files/chords_organ.wav", segment_duration=2, hop_length=128)
#     # print(features[0]["chroma"])
#     # chordlist = store_chords(features)
#     # print_all_chords(chordlist)


# if __name__ == "__main__":
#     main()