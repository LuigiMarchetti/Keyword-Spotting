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
        """
        Inicializa a classe VoiceTrainer.

        Args:
            dataset_path (str): Caminho para o diretório do dataset.
            commands (list): Lista de rótulos dos comandos.
            sample_rate (int): Taxa de amostragem (Hz).
        """
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate
        self.model = None
        self.label2idx = {label: i for i, label in enumerate(commands)}

    def preprocess_audio(self, signal, sr):
        """
        Normaliza e remove silêncios do sinal de áudio.

        Args:
            signal (np.ndarray): Sinal de áudio.
            sr (int): Taxa de amostragem.

        Returns:
            np.ndarray: Sinal pré-processado.
        """
        signal = librosa.util.normalize(signal)
        signal, _ = librosa.effects.trim(signal, top_db=20)
        return signal

    def extract_features(self, signal, sr):
        """
        Extrai os coeficientes MFCC e calcula a média.

        Args:
            signal (np.ndarray): Sinal de áudio.
            sr (int): Taxa de amostragem.

        Returns:
            np.ndarray: Vetor de 20 coeficientes MFCC médios.
        """
        n_mfcc = 20
        n_fft = 256
        hop_length = 128
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def load_data(self):
        """
        Carrega e processa os arquivos de áudio.

        Returns:
            tuple: (X, y) onde X são as características e y os rótulos.
        """
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
                    signal, sr = librosa.load(path, sr=self.sample_rate)
                    if len(signal) < 0.1 * sr:
                        continue

                    signal = self.preprocess_audio(signal, sr)
                    features = self.extract_features(signal, sr)
                    X.append(features)
                    y.append(self.label2idx[label])

                except Exception as e:
                    print(f"[ERRO] Falha ao processar {path}: {e}")
                    continue

        return np.array(X), np.array(y)

    def train(self):
        """
        Treina o modelo SVM.

        Returns:
            float: Acurácia no conjunto de teste.
        """
        X, y = self.load_data()
        if len(X) == 0:
            print("[ERRO] Nenhum dado foi carregado para treinamento!")
            return 0

        print(f"[INFO] Dados carregados: {X.shape[0]} amostras, {X.shape[1]} características")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', gamma=1000, C=1, random_state=42)
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULTADO] Acurácia no conjunto de teste: {acc:.2%}")
        print("\n[RELATÓRIO DETALHADO]")
        print(classification_report(y_test, y_pred, target_names=self.commands))
        return acc

    def save(self, model_path, label_map_path):
        """
        Salva o modelo treinado e o mapeamento de rótulos.

        Args:
            model_path (str): Caminho para salvar o modelo.
            label_map_path (str): Caminho para salvar o mapeamento.
        """
        if self.model is None:
            print("[ERRO] Modelo ainda não foi treinado!")
            return

        dump(self.model, model_path)
        with open(label_map_path, "w") as f:
            json.dump(self.label2idx, f)
        print("[INFO] Modelo e mapeamento salvos com sucesso.")

    def trainAndSave(self):
        """
        Treina e salva o modelo.
        """
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
