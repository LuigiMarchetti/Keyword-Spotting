import sounddevice as sd
import numpy as np
import librosa
import joblib
import json
import time

# Parâmetros
DURATION = 2  # segundos de escuta
SAMPLE_RATE = 11000
# COMMANDS = ["left", "right", "forward", "backward", "stop"]
COMMANDS = ["no", "yes"]

# Carregar modelo e mapeamento
model = joblib.load("SVM\\svm_model.joblib")
with open("SVM\\label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Inverter mapeamento (de índice para label)
idx_to_label = {v: k for k, v in label_mapping.items()}

# Função para extrair MFCC de um array de áudio
def extract_features_from_audio(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=55, n_fft=256, hop_length=128)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat.reshape(1, -1)

# Simulação de ação do robô
def execute_command(command):
    print(f"🔊 Comando detectado: {command}")
    if command == "yes":
        print("🤖 Escutei - 'SIM'")
    elif command == "no":
        print("🤖 Escutei - 'NÃO'")
    # if command == "left":
    #     print("🤖 Virando para a ESQUERDA")
    # elif command == "right":
    #     print("🤖 Virando para a DIREITA")
    # elif command == "forward":
    #     print("🤖 Indo PARA FRENTE")
    # elif command == "backward":
    #     print("🤖 Recuando PARA TRÁS")
    # elif command == "stop":
    #     print("⛔ Parando movimento")
    # else:
    #     print("❓ Comando não reconhecido")

# Loop principal
print("🎙️ Robô iniciado. Fale um comando: left, right, forward, backward, stop")
print("Pressione Ctrl+C para parar.\n")

try:
    while True:
        print("🎧 Escutando...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        # Transformar para vetor 1D
        audio = audio.flatten()

        # Extrair características e prever
        feat = extract_features_from_audio(audio)
        prediction = model.predict(feat)[0]
        predicted_label = idx_to_label.get(prediction, "unknown")

        execute_command(predicted_label)
        print("-" * 40)
        time.sleep(1)

except KeyboardInterrupt:
    print("🔚 Encerrando escuta...")
