import os
import librosa
import numpy as np
import json
from joblib import load
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 11000

# Carregar modelo e mapeamento
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

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=20, n_fft=256, hop_length=128)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat.reshape(1, -1)

def predict(file_path):
    features = extract_features(file_path)
    pred_idx = model.predict(features)[0]
    return idx2label[pred_idx]

def predict_with_details(file_path):
    features = extract_features(file_path)
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Probabilidades:")
    for i, prob in enumerate(probs):
        print(f"  {commands[i]}: {prob*100:.2f}%")
    return idx2label[pred_idx], probs[pred_idx]

# Testes
test_files = [
    "C:\\Projects\\Speech Emotion Recognition\\model\\no.wav",
    "C:\\Projects\\Speech Emotion Recognition\\model\\yes.wav"
]

for test_file in test_files:
    if os.path.exists(test_file):
        label, confidence = predict_with_details(test_file)
        expected = os.path.basename(os.path.dirname(test_file))
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
