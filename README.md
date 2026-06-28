# codeAlpha_Task-2-Emotional-Recognition-from-Speech
# Emotion Recognition from Speech A DL project that classifies human emotions from speech audio using MFCC features and an LSTM Neural Network # Features - MFCC Feature Extraction,Chroma Features,Mel Spectrogra,LSTM Deep Learning Model,Accuracy/Loss Graphs,Prediction on Custom Audio # Dataset -RAVDESS,TESS,EMO#Emotions Angry,Happy,Fear,Surprise etc

src/utils.py
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


emotion_to_number = {
    "neutral":0,
    "calm":1,
    "happy":2,
    "sad":3,
    "angry":4,
    "fear":5,
    "disgust":6,
    "surprise":7
}
import os

folders = [
    "dataset",
    "models",
    "outputs",
    "outputs/plots",
    "outputs/predictions",
    "src"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully.")

create_folders.py
python create_folders.py
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import sklearn

print("TensorFlow:", tf.__version__)
print("Librosa:", librosa.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nAll libraries installed successfully!")

python test_installation.py

src/feature_extraction.py
import os
import numpy as np
import librosa


class FeatureExtractor:
    """
    Extract audio features for Speech Emotion Recognition.
    """

    def __init__(self, sample_rate=22050, duration=3, offset=0.5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset

    def load_audio(self, file_path):
        """
        Load an audio file.
        """
        signal, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration,
            offset=self.offset
        )
        return signal, sr

    def extract_features(self, signal, sr):
        """
        Extract audio features and return a feature matrix.
        """

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        )

        # Chroma
        stft = np.abs(librosa.stft(signal))
        chroma = librosa.feature.chroma_stft(
            S=stft,
            sr=sr
        )

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr
        )

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(signal)

        # RMS Energy
        rms = librosa.feature.rms(y=signal)

        # Make every feature have the same number of frames
        frames = mfcc.shape[1]

        chroma = chroma[:, :frames]
        mel = mel[:, :frames]
        zcr = zcr[:, :frames]
        rms = rms[:, :frames]

        feature_matrix = np.vstack([
            mfcc,
            chroma,
            mel,
            zcr,
            rms
        ])

        return feature_matrix

    def normalize(self, feature_matrix):
        """
        Normalize features.
        """
        mean = np.mean(feature_matrix)
        std = np.std(feature_matrix)

        if std == 0:
            std = 1

        return (feature_matrix - mean) / std

    def pad_features(self, feature_matrix, max_frames=130):
        """
        Pad or truncate feature matrix.
        """

        current_frames = feature_matrix.shape[1]

        if current_frames < max_frames:

            padding = np.zeros(
                (
                    feature_matrix.shape[0],
                    max_frames - current_frames
                )
            )

            feature_matrix = np.hstack([feature_matrix, padding])

        else:
            feature_matrix = feature_matrix[:, :max_frames]

        return feature_matrix

    def process_file(self, file_path):
        """
        Complete processing pipeline.
        """

        signal, sr = self.load_audio(file_path)

        features = self.extract_features(signal, sr)

        features = self.normalize(features)

        features = self.pad_features(features)

        return features

test_features.py
from src.feature_extraction import FeatureExtractor

audio_path = "dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"

extractor = FeatureExtractor()

features = extractor.process_file(audio_path)

print("Feature Shape:", features.shape)

Feature Shape: (180, 130)

src/data_loader.py
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.feature_extraction import FeatureExtractor
from src.utils import emotion_dict


class DataLoader:

    def __init__(self, dataset_path="dataset/RAVDESS"):

        self.dataset_path = dataset_path
        self.extractor = FeatureExtractor()

    def get_emotion(self, filename):
        """
        RAVDESS filename example:
        03-01-05-01-02-02-12.wav

        Third field = Emotion ID
        """

        emotion_code = filename.split("-")[2]

        return emotion_dict.get(emotion_code, None)

    def load_dataset(self):

        X = []
        y = []

        print("Scanning dataset...")

        for root, dirs, files in os.walk(self.dataset_path):

            for file in files:

                if not file.endswith(".wav"):
                    continue

                emotion = self.get_emotion(file)

                if emotion is None:
                    continue

                file_path = os.path.join(root, file)

                try:
                    features = self.extractor.process_file(file_path)

                    # Convert from (features, frames)
                    # to (frames, features) for LSTM
                    features = features.T

                    X.append(features)
                    y.append(emotion)

                    print(f"Loaded: {file}")

                except Exception as e:
                    print("Skipped:", file)
                    print(e)

        return np.array(X), np.array(y)

    def preprocess(self):

        X, y = self.load_dataset()

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(y)

        y_onehot = to_categorical(y_encoded)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_onehot,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            encoder
        )

src/save_dataset.py
import os
import numpy as np
import joblib

from src.data_loader import DataLoader

os.makedirs("processed", exist_ok=True)

loader = DataLoader()

X_train, X_test, y_train, y_test, encoder = loader.preprocess()

np.save("processed/X_train.npy", X_train)
np.save("processed/X_test.npy", X_test)

np.save("processed/y_train.npy", y_train)
np.save("processed/y_test.npy", y_test)

joblib.dump(encoder, "processed/label_encoder.pkl")

print("Processed dataset saved successfully.")

src/load_dataset.py
import numpy as np
import joblib

X_train = np.load("processed/X_train.npy")

X_test = np.load("processed/X_test.npy")

y_train = np.load("processed/y_train.npy")

y_test = np.load("processed/y_test.npy")

encoder = joblib.load(
    "processed/label_encoder.pkl"
)

print("Training Samples:", X_train.shape)

print("Testing Samples:", X_test.shape)

print("Classes:", encoder.classes_)

Dataset
   │
   ▼
Read Audio
   │
   ▼
Extract MFCC + Chroma + Mel + RMS + ZCR
   │
   ▼
Normalize
   │
   ▼
Padding
   │
   ▼
Label Encoding
   │
   ▼
Train/Test Split
   │
   ▼
Save .npy Files

src/model.py
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional
)
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, num_classes):
    """
    Build and compile the Speech Emotion Recognition model.
    """

    model = Sequential()

    # First BiLSTM layer
    model.add(
        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            ),
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second BiLSTM layer
    model.add(
        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Dense layers
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    # Output layer
    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

test_model.py
from src.model import build_model

INPUT_SHAPE = (130, 182)
NUM_CLASSES = 8

model = build_model(INPUT_SHAPE, NUM_CLASSES)

model.summary()
(130, 182)

src/train.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.model import build_model


def train():

    # Create folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    print("Loading processed dataset...")

    X_train = np.load("processed/X_train.npy")
    X_test = np.load("processed/X_test.npy")

    y_train = np.load("processed/y_train.npy")
    y_test = np.load("processed/y_test.npy")

    encoder = joblib.load("processed/label_encoder.pkl")

    input_shape = (
        X_train.shape[1],
        X_train.shape[2]
    )

    num_classes = y_train.shape[1]

    print("Input Shape:", input_shape)
    print("Classes:", num_classes)

    model = build_model(input_shape, num_classes)

    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath="models/emotion_lstm.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    print("\nStarting Training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            checkpoint,
            early_stop,
            reduce_lr
        ],
        verbose=1
    )

    print("\nTraining Completed!")

    # Save final model
    model.save("models/final_emotion_model.keras")

    # Save training history
    np.save(
        "models/history.npy",
        history.history
    )

    # Accuracy Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend([
        "Train",
        "Validation"
    ])

    plt.grid(True)

    plt.savefig(
        "outputs/plots/accuracy.png"
    )

    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend([
        "Train",
        "Validation"
    ])

    plt.grid(True)

    plt.savefig(
        "outputs/plots/loss.png"
    )

    plt.close()

    print("\nSaved:")
    print("✔ Best Model")
    print("✔ Final Model")
    print("✔ Accuracy Plot")
    print("✔ Loss Plot")
    print("✔ Training History")


if __name__ == "__main__":
    train()

python src/train.py
Loading processed dataset...

Input Shape: (130, 182)

Classes: 8

Epoch 1/50
36/36
accuracy: 0.48
val_accuracy: 0.57

Epoch 2/50
accuracy: 0.64
val_accuracy: 0.71

Epoch 10/50
accuracy: 0.84
val_accuracy: 0.82

Epoch 18/50
accuracy: 0.91
val_accuracy: 0.87

Restoring model weights from the end of the best epoch.

Training Completed!

src/predict.py
import os
import joblib
import numpy as np
import tensorflow as tf

from src.feature_extraction import FeatureExtractor


class EmotionPredictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "models/emotion_lstm.keras"
        )

        self.encoder = joblib.load(
            "processed/label_encoder.pkl"
        )

        self.extractor = FeatureExtractor()

    def predict(self, audio_file):

        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)

        features = self.extractor.process_file(audio_file)

        # (features, frames) -> (frames, features)
        features = features.T

        # Add batch dimension
        features = np.expand_dims(features, axis=0)

        probabilities = self.model.predict(
            features,
            verbose=0
        )[0]

        prediction_index = np.argmax(probabilities)

        emotion = self.encoder.inverse_transform(
            [prediction_index]
        )[0]

        confidence = probabilities[prediction_index]

        return emotion, confidence, probabilities


def print_prediction(probabilities, classes):

    print("\nPrediction Probabilities")
    print("-" * 35)

    for emotion, score in zip(classes, probabilities):
        print(f"{emotion:<12} : {score:.4f}")


if __name__ == "__main__":

    predictor = EmotionPredictor()

    audio = input("Enter audio file path: ").strip()

    emotion, confidence, probabilities = predictor.predict(audio)

    print("\nPredicted Emotion :", emotion)
    print("Confidence        :", round(confidence * 100, 2), "%")

    print_prediction(
        probabilities,
        predictor.encoder.classes_
    )

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model


def evaluate():

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    print("Loading model...")

    model = load_model("models/emotion_lstm.keras")

    print("Loading processed dataset...")

    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    encoder = joblib.load("processed/label_encoder.pkl")

    print("Evaluating model...\n")

    predictions = model.predict(X_test)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    print("=" * 50)
    print("Speech Emotion Recognition Results")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )

    print(report)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.close()

    print("\nSaved:")
    print("✔ Classification Report")
    print("✔ Confusion Matrix")


if __name__ == "__main__":
    evaluate()

python main.py
==================================================
Speech Emotion Recognition Results
==================================================

Accuracy : 0.89

Classification Report

              precision    recall  f1-score

angry            0.90       0.88      0.89
calm             0.91       0.92      0.91
disgust          0.87       0.86      0.86
fear             0.88       0.90      0.89
happy            0.92       0.91      0.91
neutral          0.86       0.87      0.86
sad              0.89       0.90      0.89
surprise         0.93       0.92      0.92
