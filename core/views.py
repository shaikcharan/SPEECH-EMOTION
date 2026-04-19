import os
import tempfile
from pathlib import Path
import numpy as np
import librosa
import pickle
import joblib
from django.shortcuts import render
from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "speech_emotion_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0, mono=True)

        if len(audio) < sr * 3:
            audio = np.pad(audio, (0, sr * 3 - len(audio)))

        result = []

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        result.extend(np.mean(mfcc.T, axis=0))

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        result.extend(np.mean(chroma.T, axis=0))

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        result.extend(np.mean(mel.T, axis=0))

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        result.extend(np.mean(contrast.T, axis=0))

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        result.extend(np.mean(tonnetz.T, axis=0))

        return np.array(result)
    except Exception as e:
        print("Feature extraction error:", e)
        return None

def safe_inverse_transform(prediction):
    try:
        pred_index = int(prediction[0])
        return encoder.inverse_transform([pred_index])[0]
    except Exception:
        return "neutral"

def normalize_emotion_label(label):
    if not label:
        return "neutral"
    label = str(label).strip().lower()
    mapping = {
        "neutral": "neutral",
        "calm": "calm",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "anger": "angry",
        "fearful": "fearful",
        "fear": "fearful",
        "disgust": "disgust",
        "surprised": "surprised",
        "surprise": "surprised",
        "uncertain": "uncertain",
    }
    return mapping.get(label, "neutral")

def convert_audio_to_wav(input_path):
    try:
        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".wav":
            return input_path

        if ext == ".webm":
            audio = AudioSegment.from_file(input_path, format="webm")
        elif ext == ".ogg":
            audio = AudioSegment.from_file(input_path, format="ogg")
        elif ext == ".mp3":
            audio = AudioSegment.from_file(input_path, format="mp3")
        elif ext == ".m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        else:
            audio = AudioSegment.from_file(input_path)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        print("Audio conversion error:", e)
        return None

def predict_emotion(request):
    if request.method == "POST" and request.FILES.get("audio_file"):
        uploaded_file = request.FILES["audio_file"]
        temp_input_path = None
        temp_wav_path = None

        try:
            suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".webm"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_input_path = temp_file.name

            temp_wav_path = convert_audio_to_wav(temp_input_path)
            if not temp_wav_path:
                return render(request, "predict.html", {"error": "Could not process the uploaded audio file."})

            features = extract_features(temp_wav_path)
            if features is None:
                return render(request, "predict.html", {"error": "Feature extraction failed."})

            features = features.reshape(1, -1)

            expected = getattr(model, "n_features_in_", None)
            if expected and features.shape[1] != expected:
                return render(request, "predict.html", {
                    "error": f"Feature mismatch: model expects {expected}, but got {features.shape[1]}."
                })

            prediction = model.predict(features)
            raw_emotion = safe_inverse_transform(prediction)
            emotion = normalize_emotion_label(raw_emotion)

            confidence_text = "Predicted successfully"
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence_text = f"{float(np.max(probs)) * 100:.1f}% confidence"

            return render(request, "predict.html", {
                "emotion": emotion,
                "confidence": confidence_text
            })

        except Exception as e:
            return render(request, "predict.html", {
                "error": f"Prediction failed: {str(e)}"
            })

        finally:
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except Exception:
                    pass

            if temp_wav_path and os.path.exists(temp_wav_path) and temp_wav_path != temp_input_path:
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    return render(request, "predict.html")