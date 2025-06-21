import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import tempfile
import os

# Set page config to have a nice layout (must be the first Streamlit command)
st.set_page_config(page_title="Text & Voice Summarization", layout="wide")

# Lazy-load summarizer when needed (delayed loading)
summarizer = None

def load_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

def summarize_text(text):
    summarizer = load_summarizer()  # Ensure the model is loaded only when needed
    # Summarize the text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def transcribe_audio(audio_file):
    # Transcribe audio using Google Web Speech API
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #0072B5;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #1E90FF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 5px;
        padding: 10px 20px;
        margin-top: 20px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        width: 100%;
        margin-top: 10px;
    }
    .stFileUploader {
        margin-top: 20px;
    }
    .stAudio {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">Text & Voice Summarization using BERT</div>', unsafe_allow_html=True)

# Text Summarization Section
st.subheader("Text Summarization")
text_input = st.text_area("Enter text to summarize:")
if st.button("Summarize Text"):
    if text_input:
        with st.spinner('Summarizing...'):
            summary = summarize_text(text_input)
        st.success(summary)
    else:
        st.warning("Please enter some text!")

# Voice Summarization Section
st.subheader("Voice Summarization")
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    # Play the uploaded audio
    st.audio(temp_audio_path, format='audio/wav')

    # Transcribe and summarize audio when the button is pressed
    if st.button("Summarize Audio"):
        try:
            with st.spinner('Transcribing...'):
                transcribed_text = transcribe_audio(temp_audio_path)
            with st.spinner('Summarizing...'):
                summary = summarize_text(transcribed_text)
            st.success(f"Transcribed Text: {transcribed_text}")
            st.success(f"Summarized Text: {summary}")
        except Exception as e:
            st.error(f"Error processing audio: {e}")

    # Clean up temporary file after processing
    os.remove(temp_audio_path)
