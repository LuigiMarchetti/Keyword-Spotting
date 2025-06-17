import librosa
import numpy as np
import json
import os
from pathlib import Path
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações - DEVEM SER IDÊNTICAS AO TREINO
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / '../../files/models/KNN'
TEST_PATH = SCRIPT_DIR / '../../files/dataset'
TEST_PATH_RECORDED = SCRIPT_DIR / '../../files/recorded'
SAMPLES = ['ari', 'luigi']
WORDS = ['stop', 'left', 'right', 'forward', 'backward']
SAMPLE_RATE = 11000
AUDIO_LENGTH = 1.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)

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

# Load best parameters if available
best_params_path = MODEL_PATH / "best_params.json"
if best_params_path.exists():
    with open(best_params_path, "r") as f:
        best_params = json.load(f)
    print(f"Melhores parâmetros carregados: {best_params}")
else:
    print("AVISO: Parâmetros otimizados não encontrados")

# Audio preprocessing - DEVE SER IDÊNTICA AO TREINO
def preprocess_audio(signal, sr):
    """Pré-processamento do áudio"""
    # Normalização
    signal = librosa.util.normalize(signal)

    # Remoção de silêncio
    signal, _ = librosa.effects.trim(signal, top_db=20)

    # Padding/truncamento para duração consistente (1 segundo)
    target_length = sr
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')

    return signal

def extract_features(signal, sr):
    """Extrai features mais robustas do sinal de áudio - DEVE SER IDÊNTICA AO TREINO"""
    # Parâmetros melhorados para MFCC
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13

    # MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)

    # Derivadas dos MFCCs (delta e delta-delta)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Features espectrais adicionais
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]

    # Features de energia
    rms = librosa.feature.rms(y=signal)[0]

    # Estatísticas das features
    features = []

    # MFCC stats
    features.extend([mfcc.mean(axis=1), mfcc.std(axis=1),
                     mfcc.max(axis=1), mfcc.min(axis=1)])

    # Delta MFCC stats
    features.extend([mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1)])

    # Delta-delta MFCC stats
    features.extend([mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1)])

    # Spectral features stats
    features.extend([
        [spectral_centroids.mean(), spectral_centroids.std()],
        [spectral_rolloff.mean(), spectral_rolloff.std()],
        [zero_crossing_rate.mean(), zero_crossing_rate.std()],
        [rms.mean(), rms.std()]
    ])

    return np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features])

def process_audio_file(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(signal) < 0.1 * sr:
            return None
        signal = preprocess_audio(signal, sr)
        features = extract_features(signal, sr)
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

# Load model
try:
    model = load(MODEL_PATH / "knn_model.joblib")
    print("Modelo KNN carregado com sucesso!")

    # Verificar se suporta predict_proba
    if hasattr(model, 'predict_proba'):
        print("Modelo suporta probabilidades.")
    else:
        print("AVISO: Modelo não suporta probabilidades diretamente.")

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

def predict_with_details(file_path):
    features = process_audio_file(file_path)
    if features is None:
        print(f"Arquivo {os.path.basename(file_path)} muito curto - ignorando")
        return None, None

    # Reshape para predição
    features = features.reshape(1, -1)

    # Fazer predição
    predicted_idx = model.predict(features)[0]

    try:
        # Tentar obter probabilidades
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[predicted_idx]
        else:
            # Fallback: usar distâncias dos vizinhos mais próximos
            distances, indices = model.kneighbors(features)
            # Converter distâncias em pseudo-probabilidades
            max_dist = np.max(distances) + 1e-8
            pseudo_probs = (max_dist - distances.flatten()) / max_dist
            confidence = np.mean(pseudo_probs)

            # Criar array de probabilidades simuladas
            probabilities = np.zeros(len(commands))
            probabilities[predicted_idx] = confidence
            # Distribuir o resto uniformemente
            remaining = (1 - confidence) / (len(commands) - 1)
            for i in range(len(commands)):
                if i != predicted_idx:
                    probabilities[i] = remaining

    except Exception as e:
        print(f"Erro ao calcular probabilidades: {e}")
        probabilities = np.zeros(len(commands))
        probabilities[predicted_idx] = 1.0
        confidence = 1.0

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Probabilidades estimadas para cada classe:")
    for i, prob in enumerate(probabilities):
        print(f"  {commands[i]}: {prob*100:.2f}%")

    return commands[predicted_idx], confidence

# ========================
# PARTE 1 - TESTE K-FOLD
# ========================
print("\n=========== TESTE K-FOLD COM BASE DO TREINAMENTO ===========")
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

    # Fazer predições
    y_pred_indices = model.predict(X_test)
    y_pred = [commands[idx] if isinstance(idx, (int, np.integer)) else idx for idx in y_pred_indices]

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
plt.title("Matriz de Confusão - K-Fold (KNN)")
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

# Exibir informações do modelo
print(f"\n=== INFORMAÇÕES DO MODELO KNN ===")
if hasattr(model, 'named_steps'):
    # Pipeline model
    knn_step = model.named_steps.get('kneighborsclassifier')
    if knn_step:
        print(f"Número de vizinhos (k): {knn_step.n_neighbors}")
        print(f"Métrica de distância: {knn_step.metric}")
        print(f"Pesos: {knn_step.weights}")
        if hasattr(knn_step, 'p'):
            print(f"Parâmetro p (Minkowski): {knn_step.p}")
else:
    # Modelo KNN direto
    if hasattr(model, 'n_neighbors'):
        print(f"Número de vizinhos (k): {model.n_neighbors}")
        print(f"Métrica de distância: {model.metric}")
        print(f"Pesos: {model.weights}")

print("\nTeste KNN concluído!")