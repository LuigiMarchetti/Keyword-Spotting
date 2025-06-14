import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import json
from tqdm import tqdm

class VoiceTrainer:
    def __init__(self, dataset_path, commands, sample_rate=11000):
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate
        self.model = None
        self.label2idx = {label: i for i, label in enumerate(commands)}

    def extract_features(self):
        X, y = [], []
        for label in self.commands:
            folder = os.path.join(self.dataset_path, label)
            if not os.path.exists(folder):
                print(f"[ERRO] Pasta não encontrada: {folder}")
                continue
            files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            print(f"Carregando {len(files)} arquivos para '{label}'")
            for fname in tqdm(files, desc=f"Processando {label}"):
                path = os.path.join(folder, fname)
                signal, sr = librosa.load(path, sr=self.sample_rate)
                mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=20, n_fft=256, hop_length=128)
                feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
                X.append(feat)
                y.append(self.label2idx[label])
        return np.array(X), np.array(y)

    def train(self):
        X, y = self.extract_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=1e3))
        print("[INFO] Treinando modelo SVM...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULTADO] Acurácia no conjunto de teste: {acc:.2%}")
        return acc

    def save(self, model_path="svm_model.joblib", label_map_path="label_mapping.json"):
        dump(self.model, model_path)
        with open(label_map_path, "w") as f:
            json.dump(self.label2idx, f)
        print("[INFO] Modelo e mapeamento salvos com sucesso.")

# Exemplo de uso:
# trainer = VoiceTrainer("C:/Projects/Speech Emotion Recognition/files", ['yes', 'no'])
# trainer.train()
# trainer.save()
