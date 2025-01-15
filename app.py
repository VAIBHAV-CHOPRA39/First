# Import necessary libraries
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define function to process audio and generate spectrogram
def process_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Streamlit App
st.title("Audio Genre Visualization App")
st.write("Upload an audio file to generate its spectrogram.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Process audio
    st.write("Generating spectrogram...")
    spectrogram = process_audio(uploaded_file)

    # Display spectrogram
    st.write("Spectrogram:")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    st.pyplot(plt)



'''# Import necessary libraries
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained CNN model
MODEL_PATH = "model_genre_classification.h5"  # Update with your model path
model = load_model(MODEL_PATH)

# Define function to process audio and generate spectrogram
def process_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Normalize spectrogram for model input
def normalize_spectrogram(spectrogram):
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    return spectrogram / np.max(spectrogram)

# Define function to predict genre
def predict_genre(spectrogram):
    genres = ["Classical", "Jazz", "Pop", "Rock", "Hip-Hop"]  # Update with your genre labels
    prediction = model.predict(spectrogram)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

# Streamlit App
st.title("Audio Genre Classification App")
st.write("Upload an audio file to classify its genre.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Process audio
    st.write("Generating spectrogram...")
    spectrogram = process_audio(uploaded_file)

    # Display spectrogram
    st.write("Spectrogram:")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    st.pyplot(plt)
    
    # Normalize and predict genre
    st.write("Predicting genre...")
    normalized_spectrogram = normalize_spectrogram(spectrogram)
    genre = predict_genre(normalized_spectrogram)
    
    # Display prediction
    st.success(f"Predicted Genre: **{genre}**")
'''