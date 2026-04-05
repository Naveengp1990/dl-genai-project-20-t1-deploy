import streamlit as st
import torch
import numpy as np
import tempfile
import os
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load feature extractor (handles preprocessing)
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
    )
    
    # Load your fine-tuned model weights
    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10,  # CHANGE to your number of genres
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("model_fold0.pth", map_location=device))
    
    model.eval()
    model.to(device)
    return model, feature_extractor, device

# ---------------- PREDICTION ----------------
def predict(model, feature_extractor, device, audio_path):
    # Load audio
    import librosa
    audio, sample_rate = librosa.load(audio_path, sr=16000)  # AST expects 16kHz
    
    # Preprocess using HF feature extractor
    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
    
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        all_probs = probs[0].cpu().numpy()  # Get all probabilities
    
    return pred_idx, confidence, all_probs

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "flac", "ogg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("⏳ Loading model & analyzing audio..."):
        try:
            model, feature_extractor, device = load_model()
            pred_idx, confidence, all_probs = predict(model, feature_extractor, device, tmp_path)

            # TODO: REPLACE with YOUR exact genre labels (in training order!)
            GENRES = ["Blues", "Classical", "Country", "Disco", "HipHop", 
                      "Jazz", "Metal", "Pop", "Reggae", "Rock"]
            
            st.success(f"🎶 Predicted Genre: **{GENRES[pred_idx].upper()}**")
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Show all genre probabilities (FIXED)
            st.subheader("📊 All Genre Probabilities")
            probs_dict = {genre: float(prob) for genre, prob in zip(GENRES, all_probs)}
            st.bar_chart(probs_dict)
            
            # Optional: Show detailed table
            st.subheader("📋 Detailed Results")
            import pandas as pd
            df = pd.DataFrame({
                'Genre': GENRES,
                'Probability': [f"{p:.2%}" for p in all_probs]
            }).sort_values('Probability', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.write("Check that model_fold0.pth matches your training architecture.")
            import traceback
            st.code(traceback.format_exc())

    # Cleanup
    os.unlink(tmp_path)

# ---------------- SIDEBAR INFO ----------------
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
- **Model**: Audio Spectrogram Transformer (AST)
- **Input**: 16kHz audio
- **Genres**: 10 classes
""")