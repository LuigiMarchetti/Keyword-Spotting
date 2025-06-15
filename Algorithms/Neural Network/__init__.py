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
DATASET_PATH = 'C:\\Projects\\Speech Emotion Recognition\\files'
commands = ['go', 'stop', 'left', 'right', 'forward', 'backward']
SAMPLE_RATE = 11000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset personalizado
class MFCCDataset(Dataset):
    def __init__(self, files, labels, commands):
        self.files = files
        self.labels = labels
        # CORREÇÃO: usar ordem fixa dos commands
        self.label2idx = {label: i for i, label in enumerate(commands)}
        print(f"Mapeamento de labels: {self.label2idx}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20, n_fft=256, hop_length=128)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return torch.tensor(feat, dtype=torch.float32), self.label2idx[self.labels[idx]]

# Coletar caminhos e rótulos
X_paths, y_labels = [], []
for label in commands:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder):
        print(f"ERRO: Pasta {folder} não encontrada!")
        continue

    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"Encontrados {len(files)} arquivos para '{label}'")

    for fname in files:
        X_paths.append(os.path.join(folder, fname))
        y_labels.append(label)

print(f"Total de arquivos: {len(X_paths)}")
print(f"Distribuição: {dict(zip(*np.unique(y_labels, return_counts=True)))}")

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
train_dataset = MFCCDataset(X_train, y_train, commands)
test_dataset = MFCCDataset(X_test, y_test, commands)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Modelo simples MLP
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Adicionado dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = VoiceClassifier(num_classes=len(commands)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Função para calcular acurácia
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# Treinamento com mais épocas e validação
EPOCHS = 50  # Aumentado significativamente
best_acc = 0
train_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    # Calcular acurácias
    train_acc = 100. * correct / total
    val_acc = calculate_accuracy(model, test_loader)

    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Salvar melhor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "voice_model.pt")
        print(f"Novo melhor modelo salvo! Acurácia: {best_acc:.2f}%")

# Avaliação final
print(f"\nMelhor acurácia de validação: {best_acc:.2f}%")

# Salvar mapeamento de labels junto com o modelo
label_mapping = {label: i for i, label in enumerate(commands)}
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

print("Modelo e mapeamento de labels salvos!")
print(f"Mapeamento final: {label_mapping}")