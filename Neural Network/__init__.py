import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configurações
DATASET_PATH = 'C:\\Projects\\Speech Emotion Recognition\\files'
commands = ['yes']
SAMPLE_RATE = 11000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset personalizado
class MFCCDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.label2idx = {label: i for i, label in enumerate(sorted(set(labels)))}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=20, n_fft=256, hop_length=128)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return torch.tensor(feat, dtype=torch.float32), self.label2idx[self.labels[idx]]

# Coletar caminhos e rótulos
X_paths, y_labels = [], []
for label in commands:
    folder = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            X_paths.append(os.path.join(folder, fname))
            y_labels.append(label)

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_paths, y_labels, test_size=0.2, random_state=42)
train_dataset = MFCCDataset(X_train, y_train)
test_dataset = MFCCDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Modelo simples MLP
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = VoiceClassifier(num_classes=len(commands)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Treinamento com barra de progresso
EPOCHS = 1
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

# Avaliação
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
print(f"\nTest Accuracy: {100. * correct / total:.2f}%")

# Salvar modelo
torch.save(model.state_dict(), "voice_model.pt")
print("Modelo salvo como voice_model.pt")