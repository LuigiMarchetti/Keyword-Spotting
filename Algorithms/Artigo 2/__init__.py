import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
import torchvision.transforms as T
from torchvision.transforms.functional import resize

# Caminhos
DATASET_PATH = '../../files'
OUTPUT_PATH = '../../files/models/Artigo 2/'

# Comandos de voz
COMMANDS = ['backward', 'forward', 'left', 'right', 'stop']

# Hiperparâmetros
SAMPLE_RATE = 16000
IMG_SIZE = 224
N_MELS = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset PyTorch
class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelo Swin
class VoiceSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Pré-processamento
def preprocess_audio(signal, sample_rate=SAMPLE_RATE):
    signal = librosa.util.normalize(signal)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    target_length = sample_rate
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    return signal

def extract_logmel(signal, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel

def load_data(dataset_path, commands):
    label2idx = {label: i for i, label in enumerate(commands)}
    X, y = [], []

    for label in commands:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            print(f"[AVISO] Pasta não encontrada: {folder}")
            continue

        for fname in tqdm(os.listdir(folder), desc=f"Processando {label}"):
            if not fname.endswith('.wav'):
                continue
            path = os.path.join(folder, fname)
            try:
                signal, sr = librosa.load(path, sr=SAMPLE_RATE)
                if len(signal) < 0.5 * sr:
                    continue
                signal = preprocess_audio(signal)
                logmel = extract_logmel(signal)  # (128, T)

                # Redimensionar para (224, 224) e replicar para 3 canais
                img = resize(torch.tensor(logmel).unsqueeze(0), [IMG_SIZE, IMG_SIZE])  # (1, 224, 224)
                img = img.repeat(3, 1, 1)  # (3, 224, 224)

                X.append(img.numpy())
                y.append(label2idx[label])
            except Exception as e:
                print(f"[ERRO] {path}: {e}")
    return np.array(X), np.array(y), label2idx

# Função principal de treino
def train_swin():
    print(f"[INFO] Dispositivo usado: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    X, y, label2idx = load_data(DATASET_PATH, COMMANDS)

    if len(X) == 0:
        print("[ERRO] Nenhum dado carregado!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    train_ds = VoiceDataset(X_train, y_train)
    test_ds = VoiceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = VoiceSwin(num_classes=len(COMMANDS)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

    # Avaliação
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f"[RESULTADO] Acurácia de teste: {correct / total:.2%}")

    # Salvar modelo e rótulos
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'swin_model.pth'))
    with open(os.path.join(OUTPUT_PATH, 'label_mapping.json'), 'w') as f:
        json.dump(label2idx, f)
    print("[INFO] Modelo salvo com sucesso.")

if __name__ == "__main__":
    train_swin()
