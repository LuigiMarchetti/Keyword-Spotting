import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

# Configurações
DATASET_PATH = '../../files/dataset'
OUTPUT_PATH = '../../files/models/Neural Network/'
commands = ['stop', 'left', 'right', 'forward', 'backward']
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2.0  # 2 segundos
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)  # 32000 samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset simplificado
class AudioDataset(Dataset):
    def __init__(self, files, labels, commands, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment
        self.label2idx = {label: i for i, label in enumerate(commands)}
        print(f"Mapeamento de labels: {self.label2idx}")

    def __len__(self):
        return len(self.files)

    def preprocess_audio(self, signal):
        """Garante 2 segundos de áudio"""
        if len(signal) > N_SAMPLES:
            # Pega do centro
            start = (len(signal) - N_SAMPLES) // 2
            signal = signal[start:start + N_SAMPLES]
        elif len(signal) < N_SAMPLES:
            # Pad com zeros
            signal = np.pad(signal, (0, N_SAMPLES - len(signal)), mode='constant')

        # Normaliza volume
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.8

        return signal

    def add_noise(self, signal):
        """Adiciona ruído leve"""
        noise = np.random.randn(len(signal)) * 0.002
        return signal + noise

    def __getitem__(self, idx):
        signal, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        signal = self.preprocess_audio(signal)

        # Augmentation simples (só ruído)
        if self.augment and np.random.random() > 0.7:
            signal = self.add_noise(signal)

        # MFCCs básicos (13 coeficientes)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

        # Normalização simples
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        # Flatten para entrada do MLP
        mfcc_flat = mfcc.flatten()

        return torch.tensor(mfcc_flat, dtype=torch.float32), self.label2idx[self.labels[idx]]

class VoiceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, num_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(num_epochs):
        # Treinamento
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validação
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Salva melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PATH + "voice_model.pt")
            print(f"✓ Melhor modelo salvo! Acurácia: {best_acc:.2f}%")

    return best_acc

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Coletar dados
    X_paths, y_labels = [], []
    for label in commands:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            print(f"ERRO: Pasta {folder} não encontrada!")
            continue

        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"'{label}': {len(files)} arquivos")

        for fname in files:
            X_paths.append(os.path.join(folder, fname))
            y_labels.append(label)

    print(f"\nTotal: {len(X_paths)} arquivos")

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # Datasets
    train_dataset = AudioDataset(X_train, y_train, commands, augment=True)
    test_dataset = AudioDataset(X_test, y_test, commands, augment=False)

    # Descobrir tamanho da entrada (primeiro sample)
    sample_input, _ = train_dataset[0]
    input_size = sample_input.shape[0]
    print(f"Tamanho da entrada: {input_size}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modelo
    model = VoiceClassifier(input_size, len(commands)).to(device)
    print(f"Modelo criado! Device: {device}")

    # Treinamento
    print("\nIniciando treinamento...")
    best_acc = train_model(model, train_loader, test_loader)

    print(f"\nMelhor acurácia: {best_acc:.2f}%")

    # Salvar mapeamento
    label_mapping = {label: i for i, label in enumerate(commands)}
    with open(OUTPUT_PATH + "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)