<<<<<<< HEAD
# 🎵 Music Genre Classifier

A Streamlit app that uses Audio Spectrogram Transformer (AST) to classify music genres.

## Features
- Upload audio files (MP3, WAV, FLAC, OGG)
- Real-time genre prediction
- Confidence scores
- Probability distribution across all genres

## Model
- **Architecture**: Audio Spectrogram Transformer (AST)
- **Base Model**: MIT/ast-finetuned-audioset-10-10-0.4593
- **Fine-tuned on**: Custom music genre dataset
- **Number of Classes**: 10 genres

## Genres
1. Blues
2. Classical
3. Country
4. Disco
5. HipHop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/music_genre_app.git
cd music_genre_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
=======
﻿# Deep Learning and GenAI Project [BSDA2001P]
# PROJECT: Messy Mashup – Music Genre Classification
### Predicting the Right Genre from Noisy Music Mashups

### Student Name: Gnanaprakash
### Student ID: 24ds2000021

### 1. Data Input & Pre-processing

* Source Stems: The process begins with four clean `.wav` tracks (drums, bass, vocals, other) for each of the 10 genres.
 
* Standardization: Audio is resampled to a uniform 16,000 Hz (SR) and converted to mono-channel to reduce computational complexity while retaining key frequency information.
 
* Feature Extraction: Raw waveforms are converted into Log-Mel Spectrograms. This translates time-domain audio into a visual frequency-domain representation that the Transformer can "see."

### 2. Synthetic Mashup Generation (Training Phase)

* Genre-Specific Recombination: To mimic the test set, the system randomly selects stems from different songs but within the same genre (e.g., Pop Song A drums + Pop Song B vocals).

* Temporal Synchronization: Random Time-Stretching (between 0.95x and 1.05x) is applied to individual stems to simulate the rhythmic alignment artifacts mentioned in the competition brief.

* Dynamic Mixing: The four stems are summed into a single track, ensuring the model learns to identify "Jazz" even when the instrument balance (e.g., louder drums vs. quieter vocals) fluctuates.

### 3. Controlled Noise Injection

* ESC-50 Integration: Random environmental clips (rain, footsteps, sirens) from the ESC-50 dataset are overlaid onto the clean mashup.

* SNR Scaling: Noise is added at varying Signal-to-Noise Ratios (SNR) (10dB to 30dB), forcing the model to distinguish between "musical signal" and "environmental interference."

* Pitch & Gain Augmentation: Final gain adjustments and pitch shifts are applied to ensure the model doesn't overfit to specific recording volumes or keys.

### 4. The AST Classifier Architecture and CNN Architecture

* Patch Embedding: The 10-second spectrogram is broken into small, overlapping 16x16 patches, similar to how Vision Transformers process images.

* Global Self-Attention: Unlike CNNs that have a "local" view, the Audio Spectrogram Transformer (AST) uses attention mechanisms to correlate a drum beat at second 1 with a vocal melody at second 9.

* Genre Head: The Transformer's output is fed into a specialized MLP (Multi-Layer Perceptron) that maps the high-level features to one of the 10 genre probabilities.

* CNNs are effective for local patterns, but music genre is defined by long-term rhythmic and harmonic structures. The AST's self-attention mechanism allowed our model to maintain a global context of the 10-second mashup, which was crucial for ignoring localized bursts of ESC-50 noise that might have confused a CNN's local filters.

### 5. Robust Inference & Prediction

* 5-Fold Ensemble: Instead of relying on one model, the project uses 5 models trained on different data subsets. Their outputs are averaged to reduce "blind spots.

* "Test-Time Augmentation (TTA): The test file is played back at slightly different speeds (1.0, 1.02, 0.98). If a file sounds like "Rock" at all three speeds, the prediction confidence increases.

* Weighted Averaging: Chunks from the center of the audio file are given higher weight than the edges, as the middle of a song usually contains the most representative genre characteristics.

* Final Output: The system generates a submission.csv containing the id and the predicted genre based on the highest Macro F1 probability.

## Challenges: Music Genre Classification Project 

Standard models fail on this dataset because the training data (clean, individual stems) is fundamentally different from the test data (noisy, time-stretched mashups). This is a Distribution Shift problem.

## The Solution:

Our approach utilizes a State-of-the-Art Audio Spectrogram Transformer (AST). Unlike traditional CNNs that look at local textures, AST uses a Self-Attention mechanism to capture long-range global dependencies in music—allowing it to "ignore" localized noise bursts and focus on the underlying rhythmic and harmonic structure of the genre.

## Project Conclusion:

Our final model successfully bridges the gap between clean instrument stems and noisy mashups. By simulating the competition’s mixing process within our training loop and leveraging the global context of the Audio Spectrogram Transformer, we developed a system that is invariant to tempo changes, instrument re-balancing, and additive environmental noise.

### Result: 

The system provides a robust framework for real-world Music Information Retrieval (MIR) where 'clean' audio is rarely the reality.

## Project Summary:

Our system generates realistic synthetic mashups using multi-stem mixing with controlled augmentations like pitch shift, time stretch, and noise injection.
These audio signals are converted into spectrograms and passed into an Audio Spectrogram Transformer (AST), which captures both temporal and frequency dependencies using self-attention.
We improve robustness using K-fold training, label smoothing, and OneCycle learning rate scheduling.
During inference, we apply test-time augmentation, chunk-based prediction, and model ensembling to achieve high accuracy and generalization.
>>>>>>> 11a6957343327ce14780da78770eddc40c4a881a
