import os
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import json
from tqdm import tqdm

class VoiceTrainerKNN:
    def __init__(self, dataset_path, commands, sample_rate=11000):
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate
        self.model = None
        self.best_params = None
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

        # Features de energia
        rms = librosa.feature.rms(y=signal)[0]

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
            [zero_crossing_rate.mean(), zero_crossing_rate.std()],
            [rms.mean(), rms.std()]
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
            print("[INFO] Executando Grid Search para otimização de hiperparâmetros KNN...")

            # Parâmetros para busca - otimizados para KNN
            param_grid = {
                'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25],
                'kneighborsclassifier__weights': ['uniform', 'distance'],
                'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
                'kneighborsclassifier__p': [1, 2]  # Para métrica minkowski (1=manhattan, 2=euclidean)
            }

            # Pipeline com StandardScaler (importante para KNN)
            pipeline = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier()
            )

            # Grid Search
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            print(f"[INFO] Melhores parâmetros: {self.best_params}")
            print(f"[INFO] Melhor score CV: {grid_search.best_score_:.2%}")

        else:
            # Parâmetros padrão otimizados para reconhecimento de voz
            self.model = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights='distance',
                    metric='euclidean'
                )
            )
            print("[INFO] Treinando modelo KNN com parâmetros padrão...")
            self.model.fit(X_train, y_train)

        # Avaliação
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[RESULTADO] Acurácia no conjunto de teste: {acc:.2%}")
        print("\n[RELATÓRIO DETALHADO]")
        print(classification_report(y_test, y_pred, target_names=self.commands))

        return acc

    def save(self, model_path="knn_model.joblib", label_map_path="label_mapping.json",
             params_path="best_params.json"):
        if self.model is None:
            print("[ERRO] Modelo não foi treinado ainda!")
            return

        # Salvar modelo
        dump(self.model, model_path)

        # Salvar mapeamento de labels
        with open(label_map_path, "w") as f:
            json.dump(self.label2idx, f)

        # Salvar melhores parâmetros se disponíveis
        if self.best_params is not None:
            with open(params_path, "w") as f:
                json.dump(self.best_params, f, indent=2)
            print(f"[INFO] Melhores parâmetros salvos em: {params_path}")

        print(f"[INFO] Modelo salvo em: {model_path}")
        print(f"[INFO] Mapeamento de labels salvo em: {label_map_path}")

    def predict(self, audio_path):
        """Prediz a classe de um arquivo de áudio"""
        if self.model is None:
            print("[ERRO] Modelo não foi treinado ainda!")
            return None

        try:
            # Carregar e processar áudio
            signal, sr = librosa.load(audio_path, sr=self.sample_rate)
            signal = self.preprocess_audio(signal, sr)
            features = self.extract_features(signal, sr)

            # Predição
            features = features.reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Converter índice para label
            predicted_label = self.commands[prediction]
            confidence = max(probabilities)

            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_probabilities': {self.commands[i]: prob for i, prob in enumerate(probabilities)}
            }

        except Exception as e:
            print(f"[ERRO] Falha ao processar {audio_path}: {e}")
            return None

# Exemplo de uso:
if __name__ == "__main__":
    # Inicializar o treinador
    trainer = VoiceTrainerKNN(
        dataset_path="C:/Projects/Speech Emotion Recognition/files",
        commands=['go', 'stop', 'left', 'right', 'forward', 'backward']
    )

    # Treinar com Grid Search (recomendado para encontrar melhores parâmetros)
    print("=== TREINAMENTO COM GRID SEARCH ===")
    accuracy = trainer.train(use_grid_search=True)

    # Salvar modelo e parâmetros
    trainer.save()

    # Exemplo de predição (descomente se tiver um arquivo de teste)
    # result = trainer.predict("path/to/test/audio.wav")
    # if result:
    #     print(f"Predição: {result['predicted_label']} (confiança: {result['confidence']:.2%})")

    print(f"\n=== RESUMO ===")
    print(f"Acurácia final: {accuracy:.2%}")
    if trainer.best_params:
        print("Melhores parâmetros encontrados:")
        for param, value in trainer.best_params.items():
            print(f"  {param}: {value}")