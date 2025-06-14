import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import json
from tqdm import tqdm

class VoiceTrainer:
    def __init__(self, dataset_path, commands, sample_rate=16000):
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate  # Aumentado para 16kHz
        self.model = None
        self.label2idx = {label: i for i, label in enumerate(commands)}

    def extract_features(self, signal, sr):
        """Extrai features mais robustas do sinal de áudio"""
        # Parâmetros melhorados para MFCC
        n_fft = 2048
        hop_length = 512
        n_mfcc = 13

        # MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length)

        # Derivadas dos MFCCs (delta e delta-delta)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Features espectrais adicionais
        spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]

        # Estatísticas das features
        features = []

        # MFCC stats
        features.extend([mfcc.mean(axis=1), mfcc.std(axis=1),
                         mfcc.max(axis=1), mfcc.min(axis=1)])

        # Delta MFCC stats
        features.extend([mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1)])

        # Delta-delta MFCC stats
        features.extend([mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1)])

        # Spectral features stats
        features.extend([
            [spectral_centroids.mean(), spectral_centroids.std()],
            [spectral_rolloff.mean(), spectral_rolloff.std()],
            [zero_crossing_rate.mean(), zero_crossing_rate.std()]
        ])

        return np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features])

    def preprocess_audio(self, signal, sr):
        """Pré-processamento do áudio"""
        # Normalização
        signal = librosa.util.normalize(signal)

        # Remoção de silêncio
        signal, _ = librosa.effects.trim(signal, top_db=20)

        # Padding/truncamento para duração consistente (1 segundo)
        target_length = sr
        if len(signal) > target_length:
            signal = signal[:target_length]
        else:
            signal = np.pad(signal, (0, target_length - len(signal)), 'constant')

        return signal

    def load_data(self):
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
                try:
                    signal, sr = librosa.load(path, sr=self.sample_rate)

                    # Pular arquivos muito curtos
                    if len(signal) < 0.1 * sr:  # Menos que 0.1 segundo
                        continue

                    # Pré-processamento
                    signal = self.preprocess_audio(signal, sr)

                    # Extração de features
                    features = self.extract_features(signal, sr)

                    X.append(features)
                    y.append(self.label2idx[label])

                except Exception as e:
                    print(f"[ERRO] Falha ao processar {path}: {e}")
                    continue

        return np.array(X), np.array(y)

    def train(self, use_grid_search=True):
        X, y = self.load_data()

        if len(X) == 0:
            print("[ERRO] Nenhum dado foi carregado para treinamento!")
            return 0

        print(f"[INFO] Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if use_grid_search:
            print("[INFO] Executando Grid Search para otimização de hiperparâmetros...")

            # Parâmetros para busca
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svc__kernel': ['rbf', 'poly', 'sigmoid']
            }

            # Pipeline
            pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

            # Grid Search
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_

            print(f"[INFO] Melhores parâmetros: {grid_search.best_params_}")
            print(f"[INFO] Melhor score CV: {grid_search.best_score_:.2%}")

        else:
            # Parâmetros padrão melhorados
            self.model = make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
            )
            print("[INFO] Treinando modelo SVM com parâmetros padrão...")
            self.model.fit(X_train, y_train)

        # Avaliação
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[RESULTADO] Acurácia no conjunto de teste: {acc:.2%}")
        print("\n[RELATÓRIO DETALHADO]")
        print(classification_report(y_test, y_pred, target_names=self.commands))

        return acc

    def save(self, model_path="svm_model.joblib", label_map_path="label_mapping.json"):
        if self.model is None:
            print("[ERRO] Modelo não foi treinado ainda!")
            return

        dump(self.model, model_path)
        with open(label_map_path, "w") as f:
            json.dump(self.label2idx, f)
        print("[INFO] Modelo e mapeamento salvos com sucesso.")

# Exemplo de uso:
if __name__ == "__main__":
    trainer = VoiceTrainer("C:/Projects/Speech Emotion Recognition/files", ['yes', 'no'])
    trainer.train(use_grid_search=True)  # Use False para treinamento mais rápido
    trainer.save()