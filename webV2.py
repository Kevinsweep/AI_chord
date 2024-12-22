import streamlit as st
import app

st.set_page_config(page_title="SigmaChord", page_icon="🎵", layout="centered")
st.title("SigmaChord")
st.write("🎶Welcome to SigmaChord🎶")
upload_file = st.file_uploader(label="", label_visibility="collapsed", type="wav")
beat = st.number_input("beats per minute")

if st.button(label="Process Audio", type="primary"):
    if upload_file is not None:
        try:
            with st.spinner("Processing audio..."):
                loaded_model = app.load_model()
                segment_duration = 2
                if beat != 0:
                    segment_duration = 60/beat
                features = app.prepare_audio_for_chord_detection(
                    audio_path=upload_file,
                    segment_duration=segment_duration
                )
                chordlist = app.store_chords(features, loaded_model)

                st.success("Chord detection complete! 🎸")
                app.print_all_chords(chordlist)
                st.audio(upload_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a valid WAV file before processing.")

    
