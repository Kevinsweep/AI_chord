import streamlit as st
import app

upload_file = st.file_uploader(label="", label_visibility="collapsed", type="wav")

if st.button(label = "process audio", type = "primary"):
    loaded_model = app.load_model()
    features = app.prepare_audio_for_chord_detection(audio_path=upload_file, segment_duration=2)
    chordlist = app.store_chords(features, loaded_model)
    app.print_all_chords(chordlist)

    
