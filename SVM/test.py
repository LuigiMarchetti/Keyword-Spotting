import os
import librosa
import numpy as np
import json
from joblib import load

SAMPLE_RATE = 11000

model = load("svm_model.joblib")

if os.path.exists("label_mapping.json"):
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    idx2label = {v: k for k, v in label_mapping.items()}
    commands = [idx2label[i] for i in sorted(idx2label)]
    print(f"Mapeamento carregado: {label_mapping}")
else:
    commands = ['yes', 'no']
    print("AVISO: Usando mapeamento padrão.")

def extract_features(signal, sr):
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]

    features = []
    features.extend([mfcc.mean(axis=1), mfcc.std(axis=1),
                     mfcc.max(axis=1), mfcc.min(axis=1)])
    features.extend([mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1)])
    features.extend([mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1)])
    features.extend([
        [spectral_centroids.mean(), spectral_centroids.std()],
        [spectral_rolloff.mean(), spectral_rolloff.std()],
        [zero_crossing_rate.mean(), zero_crossing_rate.std()]
    ])

    return np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features])

def extract_features_from_file(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    signal = librosa.util.normalize(signal)
    signal, _ = librosa.effects.trim(signal, top_db=20)

    target_length = sr
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')

    features = extract_features(signal, sr)
    return features.reshape(1, -1)

def predict(file_path):
    features = extract_features_from_file(file_path)
    pred_idx = model.predict(features)[0]
    return idx2label[pred_idx]

def predict_with_details(file_path):
    features = extract_features_from_file(file_path)
    pred_idx = model.predict(features)[0]

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Predição (sem probabilidades disponíveis)")

    confidence = 1.0
    if hasattr(model, 'decision_function'):
        try:
            decision = model.decision_function(features)[0]
            if isinstance(decision, np.ndarray):
                confidence = np.max(np.abs(decision)) / (np.sum(np.abs(decision)) + 1e-8)
            else:
                confidence = 1 / (1 + np.exp(-np.abs(decision)))
            print(f"Confiança estimada: {confidence*100:.2f}%")
        except:
            print("Confiança: não disponível")

    return idx2label[pred_idx], confidence

test_files = [
    "C:\\Projects\\Speech Emotion Recognition\\model\\no.wav",
    "C:\\Projects\\Speech Emotion Recognition\\model\\yes.wav"
]

for test_file in test_files:
    if os.path.exists(test_file):
        label, confidence = predict_with_details(test_file)
        expected = os.path.splitext(os.path.basename(test_file))[0]
        print(f"Predição: {label} (confiança: {confidence*100:.2f}%)")
        print(f"Esperado: {expected}")
        print(f"Correto: {'✓' if label == expected else '✗'}")
        print("-" * 50)
    else:
        print(f"Arquivo não encontrado: {test_file}")

if len(test_files) > 0 and os.path.exists(test_files[0]):
    print("\nTeste rápido:")
    label = predict(test_files[0])
    print(f"Predição: {label}")