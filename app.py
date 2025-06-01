
import streamlit as st
import joblib
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
from pydub import AudioSegment
import tempfile, os

st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎤 Accent Classifier App")
st.markdown("Upload an audio file and I'll predict the speaker's accent!")

@st.cache_resource
def load_model():
    return joblib.load("accent_classifier_model.pkl")

@st.cache_resource
def load_speechbrain_classifier():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

loaded_model = load_model()
classifier = load_speechbrain_classifier()

def convert_to_wav(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        audio = AudioSegment.from_file(uploaded_file)
        audio.export(tmp_wav.name, format="wav")
        return tmp_wav.name

def predict_accent(file_path):
    signal, fs = torchaudio.load(file_path)

    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
        fs = 16000

    if signal.ndim > 1:
        signal = signal.mean(dim=0).unsqueeze(0)

    with torch.no_grad():
        emb = classifier.encode_batch(signal)
        emb_np = emb.squeeze().cpu().numpy().reshape(1, -1)
        pred = loaded_model.predict(emb_np)[0]
        probs = loaded_model.predict_proba(emb_np)[0]
        confidence = np.max(probs)
        return pred, confidence, signal.squeeze().numpy(), fs

uploaded_file = st.file_uploader("Upload audio (.wav, .mp3, .flac, .mp4)", type=["wav", "mp3", "flac", "mp4"])

if uploaded_file:
    st.audio(uploaded_file)

    wav_path = convert_to_wav(uploaded_file)

    with st.spinner("Analyzing..."):
        try:
            pred_accent, conf, signal_np, fs = predict_accent(wav_path)

            # Plot waveform
            st.markdown("### 📊 Waveform")
            fig1, ax1 = plt.subplots(figsize=(10, 2))
            ax1.plot(np.linspace(0, len(signal_np) / fs, len(signal_np)), signal_np)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Waveform")
            st.pyplot(fig1)

            # Plot spectrogram
            st.markdown("### 🌈 Spectrogram")
            spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=fs)(torch.tensor(signal_np).unsqueeze(0))
            db_spec = torchaudio.transforms.AmplitudeToDB()(spectrogram)
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            im = ax2.imshow(db_spec[0].numpy(), origin='lower', aspect='auto', cmap='inferno')
            fig2.colorbar(im, ax=ax2, format="%+2.0f dB")
            ax2.set_title("Mel Spectrogram")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Mel Bin")
            st.pyplot(fig2)

            # Prediction
            st.success(f"🎯 Accent: **{pred_accent}**")
            st.progress(int(conf * 100))
            st.write(f"🧠 Confidence: **{conf * 100:.2f}%**")

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            os.remove(wav_path)
