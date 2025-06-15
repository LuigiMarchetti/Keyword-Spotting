import torch
import librosa
import numpy as np
import torch.nn as nn
import json
import os

# Classe do modelo deve ser igual à usada no treino
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
                nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Parâmetros
SAMPLE_RATE = 11000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CORREÇÃO: Carregar mapeamento de labels do arquivo
if os.path.exists("label_mapping.json"):
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    # Criar lista de commands na ordem correta
    commands = [None] * len(label_mapping)
    for label, idx in label_mapping.items():
        commands[idx] = label
    print(f"Mapeamento carregado: {label_mapping}")
else:
    # Fallback para compatibilidade
    commands = ['yes', 'no', 'go', 'stop', 'left', 'right', 'forward', 'backward']
    print("AVISO: Usando mapeamento padrão. Recomendo retreinar o modelo com o código corrigido.")

# Criar modelo e carregar pesos
model = VoiceClassifier(num_classes=len(commands))
model.load_state_dict(torch.load("voice_model.pt", map_location=device))
model.to(device)
model.eval()

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20, n_fft=256, hop_length=128)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return torch.tensor(feat, dtype=torch.float32)

def predict(file_path):
    features = extract_features(file_path).to(device)
    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
    return commands[predicted_idx], probs[0][predicted_idx].item()

def predict_with_details(file_path):
    """Versão mais detalhada para debug"""
    features = extract_features(file_path).to(device)
    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print(f"Logits brutos: {outputs[0].cpu().numpy()}")
    print("Probabilidades:")
    for i, (command, prob) in enumerate(zip(commands, probs[0])):
        print(f"  {command}: {prob.item()*100:.2f}%")

    predicted_idx = torch.argmax(probs, dim=1).item()
    return commands[predicted_idx], probs[0][predicted_idx].item()

# Teste com múltiplos arquivos para verificar
test_files = [
    "C:\\Projects\\Speech Emotion Recognition\\recorded\\no.wav",
    "C:\\Projects\\Speech Emotion Recognition\\recorded\\yes.wav"
    #"C:\\Projects\\Speech Emotion Recognition\\files\\no\\0a2b400e_nohash_1.wav",
    #"C:\\Projects\\Speech Emotion Recognition\\files\\yes\\0a7c2a8d_nohash_0.wav"  # adicione um arquivo yes
]

for test_file in test_files:
    if os.path.exists(test_file):
        label, confidence = predict_with_details(test_file)
        expected = os.path.basename(os.path.dirname(test_file))  # pega o nome da pasta
        print(f"Predição: {label} (confiança: {confidence*100:.2f}%)")
        print(f"Esperado: {expected}")
        print(f"Correto: {'✓' if label == expected else '✗'}")
        print("-" * 50)
    else:
        print(f"Arquivo não encontrado: {test_file}")

# Teste simples
if len(test_files) > 0 and os.path.exists(test_files[0]):
    print("\nTeste rápido:")
    label, confidence = predict(test_files[0])
    print(f"Predição: {label} com confiança de {confidence*100:.2f}%")