import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms.functional import resize
from tqdm import tqdm

# Caminhos
TEST_PATH = '../../files'  # mesma estrutura do treinamento
MODEL_PATH = '../../files/models/Artigo 2/swin_model.pth'
LABEL_PATH = '../../files/models/Artigo 2/label_mapping.json'

# Hiperparâmetros
SAMPLE_RATE = 16000
IMG_SIZE = 224
N_MELS = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modelo
class VoiceSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Pré-processamento
def preprocess_audio(signal):
    signal = librosa.util.normalize(signal)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    target_length = SAMPLE_RATE
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    return signal

def extract_logmel(signal, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel

def load_test_data(test_path, label2idx):
    X = []
    y = []
    for label, idx in label2idx.items():
        folder = os.path.join(test_path, label)
        if not os.path.exists(folder):
            continue

        for fname in tqdm(os.listdir(folder), desc=f"Testando {label}"):
            if not fname.endswith('.wav'):
                continue
            path = os.path.join(folder, fname)
            try:
                signal, _ = librosa.load(path, sr=SAMPLE_RATE)
                if len(signal) < 0.5 * SAMPLE_RATE:
                    continue
                signal = preprocess_audio(signal)
                logmel = extract_logmel(signal)

                img = resize(torch.tensor(logmel).unsqueeze(0), [IMG_SIZE, IMG_SIZE])
                img = img.repeat(3, 1, 1)

                X.append(img.unsqueeze(0))
                y.append(idx)
            except Exception as e:
                print(f"[ERRO] {path}: {e}")
    return X, y

def test():
    with open(LABEL_PATH) as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    model = VoiceSwin(num_classes=len(label2idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    X, y_true = load_test_data(TEST_PATH, label2idx)

    y_pred = []
    with torch.no_grad():
        for xb in X:
            xb = xb.to(DEVICE, dtype=torch.float32)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).item()
            y_pred.append(pred)

    # Métricas
    print(f"\n[ACURÁCIA]: {(np.array(y_pred) == np.array(y_true)).mean():.2%}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))

    print("[MATRIZ DE CONFUSÃO]")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test()
