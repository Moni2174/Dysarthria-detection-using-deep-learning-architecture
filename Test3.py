import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 📂 Paths
MODEL_PATH = r"C:\Users\monis\PycharmProjects\Dysarthia\dysarthria_cnn_bilstm_balanced.h5"
TEST_AUDIO_FILE = r"C:\Users\monis\Documents\Dysarthia\Pre-process\F_ND\wav_arrayMic_FC01S01_0089.wav"
RESULTS_FOLDER = r"C:\Users\monis\Documents\Dysarthia\Test\Results"
FEATURES_CSV = r"C:\Users\monis\Documents\Dysarthia\features.csv"

# 🔹 Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# 🔹 Detect Model Input Shape
input_shape = model.input_shape
if len(input_shape) == 3:  # Expected Shape: (None, time_steps, features)
    _, expected_time_steps, expected_features = input_shape
    print(f"✅ Model expects (time_steps, features) = ({expected_time_steps}, {expected_features})")
elif len(input_shape) == 4:  # In case there's a channel dimension
    _, expected_time_steps, expected_features, _ = input_shape
    print(f"✅ Model expects (time_steps, features, channels) = ({expected_time_steps}, {expected_features}, 1)")
else:
    raise ValueError(f"❌ Unexpected Model Input Shape: {input_shape}")

# 🔹 Extract Label Names from CSV
df = pd.read_csv(FEATURES_CSV)
unique_labels = df["Label"].unique()
label_mapping = {i: label for i, label in enumerate(unique_labels)}
print(f"📌 Label Mapping: {label_mapping}")

# 🎤 **Feature Extraction Function**
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        # Extract multiple features
        features = []
        features.append(np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40), axis=1))
        features.append(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1))
        features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr), axis=1))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr), axis=1))
        features.append(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1))
        features.append(np.mean(librosa.feature.zero_crossing_rate(y=audio), axis=1))
        features.append(np.mean(librosa.feature.rms(y=audio), axis=1))

        features = np.concatenate(features, axis=0)  # Flatten feature vector

        # Ensure correct feature size
        if features.shape[0] < expected_features:
            features = np.pad(features, (0, expected_features - features.shape[0]), mode='constant')
        else:
            features = features[:expected_features]

        # Normalize using StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features.reshape(1, -1))

        # ✅ **Fix Input Shape** → Ensure (time_steps, features)
        features = np.tile(features, (expected_time_steps, 1))  # Repeat to match time steps

        return audio, sr, features
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None, None, None

# 🚀 **Prediction Function**
def predict_dysarthria(audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None, None, None

    print(f"\n🔍 Processing: {audio_file}")
    audio, sr, features = extract_features(audio_file)
    if features is None:
        return None, None, None

    # ✅ **Fix Input Shape for CNN+LSTM**
    features = np.expand_dims(features, axis=0)  # Add batch dim (1, time_steps, features)

    print(f"✅ Input Shape Before Prediction: {features.shape}")  # Should match model's expected shape

    # 🚀 Make Prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    full_label = label_mapping.get(predicted_class, "Unknown")

    # 🔹 Convert to Dysarthria or Non-Dysarthria
    if "WD" in full_label:  # Check if label contains "WD" (With Dysarthria)
        result = "Dysarthric"
    elif "ND" in full_label:  # Check if label contains "ND" (Non-Dysarthric)
        result = "Non-Dysarthric"
    else:
        result = "Unknown"

    confidence = prediction[0][predicted_class]
    print(f"🎯 **Predicted Class:** {full_label}")
    print(f"🔹 **Dysarthria Status:** {result} (Confidence: {confidence:.4f})")
    print(f"🔹 Raw Prediction Probabilities: {prediction}")

    return audio, sr, result

# 🎨 **Visualization Function**
def plot_results(audio, sr, result):
    if audio is None:
        print("⚠️ No audio data available for visualization.")
        return

    plt.figure(figsize=(12, 8))

    # 🔹 Waveform
    plt.subplot(3, 1, 1)
    plt.plot(audio, color='blue')
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # 🔹 Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (Predicted: {result})")

    # 🔹 MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, x_axis="time", cmap="viridis")
    plt.colorbar()
    plt.title("MFCCs")

    plt.tight_layout()
    plt.show()

# 🏃 **Run Prediction**
audio, sr, result = predict_dysarthria(TEST_AUDIO_FILE)
plot_results(audio, sr, result)
