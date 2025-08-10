import os
import numpy as np
import librosa
import soundfile as sf
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuration
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",

}
TARGET_EMOTIONS = {"calm", "happy", "fearful", "disgust"}
RAVDESS_PATH = "/Users/devashreepk/PycharmProjects/Machine-learning/SER project/RAVDESS_Data"




#Feature Extraction
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_path) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            result = np.hstack((result, np.mean(mfccs.T, axis=0)))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            result = np.hstack((result, np.mean(chroma.T, axis=0)))
        if mel:
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            result = np.hstack((result, np.mean(mel.T, axis=0)))

    return result

#except Exception as e:
 #   print(f"[Error] Failed to process {file_path}: {e}")
  #  return None

# Load Data
def load_data():
    X, y = [], []
    for root, _, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
#Extract emotion label from filename
                parts = file.split("-")
                if len(parts) < 3:
                    continue #skip malformed filename
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion in TARGET_EMOTIONS:
                    #file_path = os.path.join(root, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)
    return np.array(X), np.array(y)

# Example usage
print("[INFO] Loading data and extracting features...")
X, y = load_data()
print(f"[INFO] Feature extraction complete. Total samples: {len(X)}")

# Train Model
def train_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, max_iter=500)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy: {:.2f}%".format(acc * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=list(TARGET_EMOTIONS))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_EMOTIONS, yticklabels=TARGET_EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
# Main Program
if __name__ == "__main__":
    print("\n[INFO] Loading data and extracting features...")
    X, y = load_data()
    print("[INFO] Feature extraction complete. Total samples:", len(X))

    print("\n[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("\n[INFO] Training model...")
    model = train_model(X_train, y_train)

    print("\n[INFO] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\n[INFO] Done.")
