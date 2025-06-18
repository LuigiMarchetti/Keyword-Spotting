import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import json
from tqdm import tqdm

class VoiceTrainer:
    def __init__(self, dataset_path, commands, sample_rate=11000):
        """Inicializa a classe VoiceTrainer."""
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate
        self.model = None
        self.label2idx = {label: i for i, label in enumerate(commands)}

    def preprocess_audio(self, signal, sr):
        """Normaliza e remove silêncios do sinal de áudio, retorna o sinal pré-processado."""
        # Normalização do sinal
        signal = librosa.util.normalize(signal)
        # Remoção de silêncios
        signal, _ = librosa.effects.trim(signal, top_db=20)
        return signal

    def extract_features(self, signal, sr):
        """Extrai 20 MFCCs médios do sinal de áudio."""
        n_mfcc = 20
        n_fft = 256
        hop_length = 128
        # Extração dos coeficientes MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # Calcula a média dos MFCCs
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def load_data(self):
        """Carrega e processa os arquivos de áudio do dataset."""
        X, y = [] ,[]
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
                    # Carrega o arquivo de áudio
                    signal, sr = librosa.load(path, sr=self.sample_rate)
                    # Ignora arquivos muito curtos
                    if len(signal) < 0.1 * sr:
                        continue
                    # Pré-processamento do áudio
                    signal = self.preprocess_audio(signal, sr)
                    # Extração de features
                    features = self.extract_features(signal, sr)
                    X.append(features)
                    y.append(self.label2idx[label])
                except Exception as e:
                    print(f"[ERRO] Falha ao processar {path}: {e}")
                    continue

        return np.array(X), np.array(y)

    def train(self):
        """Treina o modelo SVM e avalia no conjunto de teste."""
        # Carrega e processa os dados
        X, y = self.load_data()
        if len(X) == 0:
            print("[ERRO] Nenhum dado foi carregado para treinamento!")
            return 0

        print(f"[INFO] Dados carregados: {X.shape[0]} amostras, {X.shape[1]} características")
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Cria pipeline com normalização e SVM
        self.model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', gamma=1000, C=1, random_state=42)
        )
        # Treina o modelo
        self.model.fit(X_train, y_train)

        # Predição no conjunto de teste
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULTADO] Acurácia no conjunto de teste: {acc:.2%}")
        print("\n[RELATÓRIO DETALHADO]")
        print(classification_report(y_test, y_pred, target_names=self.commands))
        return acc

    def save(self, model_path, label_map_path):
        """Salva o modelo treinado e o mapeamento de rótulos."""
        if self.model is None:
            print("[ERRO] Modelo ainda não foi treinado!")
            return
        # Salva o modelo treinado e labels
        dump(self.model, model_path)
        with open(label_map_path, "w") as f:
            json.dump(self.label2idx, f)
        print("[INFO] Modelo e mapeamento salvos com sucesso.")

    def trainAndSave(self):
        """Treina e salva o modelo SVM."""
        acc = self.train()
        if acc:
            model_path = "../../files/models/Artigo 1/"
            self.save(model_path + "svm_model.joblib", model_path + "label_mapping.json")

if __name__ == "__main__":
    dataset_path = "../../files/dataset"
    trainer = VoiceTrainer(
        dataset_path,
        ['backward', 'forward', 'left', 'right', 'stop']
    )
    trainer.trainAndSave()
