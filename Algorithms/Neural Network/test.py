import librosa
import numpy as np
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações - DEVEM SER IDÊNTICAS AO TREINO
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / '../../files/models/Neural Network'
TEST_PATH = SCRIPT_DIR / '../../files/dataset'
TEST_PATH_RECORDED = SCRIPT_DIR / '../../files/recorded'
SAMPLES = ['ari', 'luigi']
WORDS = ['stop', 'left', 'right', 'forward', 'backward']
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definição do modelo - DEVE SER IDÊNTICA AO TREINO
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

# Load label mapping
label_mapping_path = MODEL_PATH / "label_mapping.json"
if label_mapping_path.exists():
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
    commands = [None] * len(label_mapping)
    for label, idx in label_mapping.items():
        commands[idx] = label
    print(f"Mapeamento carregado: {label_mapping}")
else:
    commands = ['stop', 'left', 'right', 'forward', 'backward']
    print("AVISO: Usando mapeamento padrão")

# Audio preprocessing - DEVE SER IDÊNTICA AO TREINO
def preprocess_audio(signal):
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

def extract_features(signal, sr):
    """Extrai MFCCs - DEVE SER IDÊNTICA AO TREINO"""
    # MFCCs básicos (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    # Normalização simples
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    # Flatten para entrada do MLP
    mfcc_flat = mfcc.flatten()

    return mfcc_flat

def process_audio_file(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(signal) < 0.1 * sr:
            return None
        signal = preprocess_audio(signal)
        features = extract_features(signal, sr)
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

# Load model
try:
    # Primeiro, precisamos descobrir o tamanho da entrada
    # Carregamos um arquivo de exemplo para isso
    sample_file = None
    for label in WORDS:
        dir_path = TEST_PATH / label
        if dir_path.exists():
            wav_files = list(dir_path.glob("*.wav"))
            if wav_files:
                sample_file = wav_files[0]
                break

    if sample_file is None:
        print("ERRO: Nenhum arquivo de exemplo encontrado para determinar o tamanho da entrada!")
        exit()

    sample_features = process_audio_file(sample_file)
    if sample_features is None:
        print("ERRO: Não foi possível processar arquivo de exemplo!")
        exit()

    input_size = len(sample_features)
    print(f"Tamanho da entrada detectado: {input_size}")

    # Criar e carregar modelo
    model = VoiceClassifier(input_size, len(commands)).to(device)
    model_path = MODEL_PATH / "voice_model.pt"

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Modo de avaliação
        print("Modelo de rede neural carregado com sucesso!")
    else:
        print(f"ERRO: Arquivo do modelo não encontrado: {model_path}")
        exit()

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

def predict_with_details(file_path):
    features = process_audio_file(file_path)
    if features is None:
        print(f"Arquivo {os.path.basename(file_path)} muito curto - ignorando")
        return None, None

    # Converter para tensor e fazer predição
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(outputs, dim=1)[0].item()
        confidence = probabilities[predicted_idx].item()

        # Converter probabilidades para numpy para compatibilidade
        probabilities_np = probabilities.cpu().numpy()

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Probabilidades para cada classe:")
    for i, prob in enumerate(probabilities_np):
        print(f"  {commands[i]}: {prob*100:.2f}%")

    return commands[predicted_idx], confidence

# ========================
# PARTE 1 - TESTE K-FOLD
# ========================
print(f"\n=========== TESTE K-FOLD COM BASE DO TREINAMENTO ===========")
print(f"Device utilizado: {device}")

X_kf = []
y_kf = []
for label in WORDS:
    dir_path = TEST_PATH / label
    if not dir_path.exists():
        print(f"Diretório não encontrado: {dir_path}")
        continue

    files_processed = 0
    for file in dir_path.glob("*.wav"):
        features = process_audio_file(file)
        if features is not None:
            X_kf.append(features)
            y_kf.append(label)
            files_processed += 1

    print(f"Processados {files_processed} arquivos de '{label}'")

X_kf = np.array(X_kf)
y_kf = np.array(y_kf)
print(f"Total de amostras: {len(y_kf)}")

if len(y_kf) == 0:
    print("ERRO: Nenhuma amostra foi processada!")
    exit()

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all = []
y_pred_all = []

for i, (train_idx, test_idx) in enumerate(kf.split(X_kf, y_kf), 1):
    print(f"\n--- Fold {i} ---")
    X_test = X_kf[test_idx]
    y_test = y_kf[test_idx]

    # Converter para tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted_indices = torch.max(outputs, 1)
        predicted_indices = predicted_indices.cpu().numpy()

    y_pred = [commands[idx] for idx in predicted_indices]

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    # Acurácia do fold
    fold_acc = sum(1 for true, pred in zip(y_test, y_pred) if true == pred) / len(y_test)
    print(f"Acurácia do Fold {i}: {fold_acc*100:.2f}%")

print("\n=== RELATÓRIO GERAL - K-FOLD ===")
print(classification_report(y_true_all, y_pred_all, labels=WORDS))

# Matriz de confusão
cm = confusion_matrix(y_true_all, y_pred_all, labels=WORDS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=WORDS,
            yticklabels=WORDS, cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - K-Fold (Rede Neural)")
plt.tight_layout()
plt.show()

# ========================
# PARTE 2 - TESTE CASEIRO
# ========================
print("\n=========== TESTE COM AMOSTRAS CASEIRAS ===========")
total_correct = 0
total_files = 0

for speaker in SAMPLES:
    speaker_dir = TEST_PATH_RECORDED / speaker
    if not speaker_dir.exists():
        print(f"Diretório não encontrado: {speaker_dir}")
        continue

    print(f"\n=== Testando arquivos de {speaker} ===")
    wav_files = [f for f in speaker_dir.glob('*.wav')]
    speaker_correct = 0
    valid_files = 0

    for wav_file in wav_files:
        expected_label = wav_file.stem.split('_')[0]
        if expected_label not in [cmd for cmd in commands if cmd]:
            print(f"Pulando {wav_file.name} - label '{expected_label}' não foi treinado")
            continue

        try:
            label, confidence = predict_with_details(wav_file)
            if label is None:
                continue

            is_correct = label == expected_label
            if is_correct:
                speaker_correct += 1
                total_correct += 1

            total_files += 1
            valid_files += 1

            print(f"Predição: {label} ({confidence*100:.2f}%) | Esperado: {expected_label} {'✓' if is_correct else '✗'}")

        except Exception as e:
            print(f"Erro em {wav_file.name}: {str(e)}")

    if valid_files > 0:
        print(f"Acurácia para {speaker}: {speaker_correct}/{valid_files} = {speaker_correct/valid_files*100:.2f}%")
    else:
        print(f"Nenhum arquivo válido para {speaker}")

if total_files > 0:
    print(f"\n=== RESULTADO FINAL - CASEIRO ===")
    print(f"Acurácia geral: {total_correct}/{total_files} = {total_correct/total_files*100:.2f}%")
else:
    print("Nenhum arquivo válido testado!")

print(f"\nTeste concluído! Device utilizado: {device}")