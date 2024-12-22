# SigmaChord

SigmaChord is a chord detection web app that uses AI to identify chords from wav files. It uses an LSTM model trained on the CQT Chroma of guitar chords on the C major scale. The app is built using the streamlit python library. 

# Functions

## app.py

### prepare_audio_for_chord_detection(audio_path = "test.wav", segment_duration=2.0, hop_length=128, sr = 10000)

#### Description:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a function to separate wav files into multiple parts and convert them into chromagrams.

#### Parameters:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;audio_path: path to wav file  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;segment_duration: length of each chroma in seconds.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hop_length: one of the parameters of the librosa function to extract cqt chroma (do not change value)  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sr: one of the parameters of the librosa function to extract cqt chroma (do not change value)

#### Return:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a list of chromagrams from the sound
  
### store_chords(features, loaded_model)

#### Description:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a function to predict chords from previously extracted chromagrams

#### Parameters:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: the chromas that want to be analyzed  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loaded_model: the h5 AI model

#### Return:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a list of the chords in the wav file

### print_all_chords(chordlist)  

#### Description:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a function to print all the chords predicted by the AI

#### Parameters:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;chordlist: list of chords returned by store_chords 

#### Return:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-

### load_model()

#### Description:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a function to load model. It is called everytime the web app is run

#### Parameters:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;path_to_model: path to h5 model

#### Return:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-
