import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Emotions mapping from RAVDESS filename codes
EMOTIONS = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0, mono=True)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms.T, axis=0)

        return np.hstack([
            mfcc_mean,
            chroma_mean,
            mel_mean,
            contrast_mean,
            zcr_mean,
            rms_mean
        ])
    except Exception as e:
        print(f"Feature extraction error for {file_path}: {e}")
        return None

def preprocess_dataset(dataset_path):
    features = []
    labels = []

    print("Processing RAVDESS dataset...")

    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                parts = file.split('-')

                try:
                    emotion_code = int(parts[2])
                except Exception:
                    continue

                if emotion_code in EMOTIONS:
                    emotion = EMOTIONS[emotion_code]
                    feature_vector = extract_features(file_path)

                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(emotion)
                        print(f"Processed: {file} -> {emotion}")

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    dataset_path = "dataset"

    print("=== RAVDESS Speech Emotion Dataset Preprocessing ===")
    X, y = preprocess_dataset(dataset_path)

    print("\nDataset Summary:")
    print(f"Total files processed: {len(X)}")
    print(f"Features shape: {X.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\nSaved:")
    print("- X_train.npy, X_test.npy")
    print("- y_train.npy, y_test.npy")
    print("- label_encoder.pkl")
    print("\nPreprocessing complete! Ready for model training.")