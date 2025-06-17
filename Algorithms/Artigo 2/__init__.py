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
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights

DATASET_PATH = '../../files'

class VoiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class VoiceSwin(nn.Module):
    def __init__(self, num_classes):
        super(VoiceSwin, self).__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class VoiceTrainerSwin:
    OUTPUT_PATH = '../../files/models/Artigo 2/'

    def __init__(self, dataset_path, commands, sample_rate=16000, device=None):
        self.dataset_path = dataset_path
        self.commands = commands
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label2idx = {label: i for i, label in enumerate(commands)}
        print(f"[INFO] Usando dispositivo: {self.device}")
        if self.device == "cuda":
            print(f"[INFO] Nome da GPU: {torch.cuda.get_device_name(0)}")

    def preprocess_audio(self, signal):
        signal = librosa.util.normalize(signal)
        signal, _ = librosa.effects.trim(signal, top_db=20)
        target_length = self.sample_rate
        if len(signal) > target_length:
            signal = signal[:target_length]
        else:
            signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
        return signal

    def extract_logmel(self, signal):
        mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_mels=128)
        logmel = librosa.power_to_db(mel, ref=np.max)
        return logmel

    def load_data(self):
        X, y = [], []
        for label in self.commands:
            folder = os.path.join(self.dataset_path, label)
            if not os.path.exists(folder):
                print(f"[ERRO] Pasta não encontrada: {folder}")
                continue
            for fname in tqdm(os.listdir(folder), desc=f"Processando {label}"):
                if not fname.endswith('.wav'):
                    continue
                path = os.path.join(folder, fname)
                try:
                    signal, sr = librosa.load(path, sr=self.sample_rate)
                    if len(signal) < 0.5 * sr:
                        continue
                    signal = self.preprocess_audio(signal)
                    logmel = self.extract_logmel(signal)
                    logmel = np.stack([logmel] * 3, axis=0)  # (3, H, W)
                    X.append(logmel)
                    y.append(self.label2idx[label])
                except Exception as e:
                    print(f"Erro ao processar {path}: {e}")
        return np.array(X), np.array(y)

    def train(self, epochs=30, batch_size=16, lr=1e-4):
        X, y = self.load_data()
        if len(X) == 0:
            print("[ERRO] Nenhum dado carregado!")
            return

        max_w = max(x.shape[2] for x in X)
        X = np.array([np.pad(x, ((0, 0), (0, 0), (0, max_w - x.shape[2])), mode='constant') for x in X])

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        train_ds = VoiceDataset(X_train, y_train)
        test_ds = VoiceDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        num_classes = len(self.commands)
        self.model = VoiceSwin(num_classes).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

        # Avaliação
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        print(f"\n[RESULTADO] Acurácia no teste: {correct / total:.2%}")

    def save(self, model_name='swin_model.pth', label_map_name='label_mapping.json'):
        model_path = os.path.join(self.OUTPUT_PATH, model_name)
        label_map_path = os.path.join(self.OUTPUT_PATH, label_map_name)

        torch.save(self.model.state_dict(), model_path)
        with open(label_map_path, 'w') as f:
            json.dump(self.label2idx, f)
        print("[INFO] Modelo e mapeamento salvos.")

    def trainAndSave(self):
        self.train()
        self.save()

if __name__ == "__main__":
    trainer = VoiceTrainerSwin(
        DATASET_PATH,
        ['backward', 'forward', 'left', 'right', 'stop']
    )
    trainer.trainAndSave()
