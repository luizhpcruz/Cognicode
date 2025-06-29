import os
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt

# Importações do Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram

def bandpass_filter(sig, low=1, high=50, fs=256):
    """Aplica um filtro passa-banda em um sinal."""
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def map_signal_to_qubit_rotation(value: float, min_val: float, max_val: float) -> float:
    """Mapeia um valor de sinal para um ângulo de rotação Ry (0 a pi)."""
    normalized_val = (value - min_val) / (max_val - min_val)
    return normalized_val * np.pi

def create_quantum_symbolic_tape(signal: np.ndarray, num_qubits: int, noise_level: float = 0.01) -> list:
    """
    Cria uma fita simbólica quântica a partir de um sinal clássico.
    Cada valor do sinal é mapeado para a rotação de um qubit, com ruído.
    """
    min_val, max_val = signal.min(), signal.max()
    tape_symbols = []

    # 1. Crie um modelo de ruído (simula um canal de comunicação ruidoso)
    error = depolarizing_error(noise_level, num_qubits)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'rz', 'ry', 'measure'], [0, 1])

    # 2. Crie o simulador com o modelo de ruído
    simulator = AerSimulator(noise_model=noise_model)

    # 3. Processe o sinal em blocos (a cada 100 amostras, por exemplo)
    for i in range(0, len(signal), 100):
        block_signal = signal[i:i+100]
        if len(block_signal) == 0: continue
        avg_val = np.mean(block_signal)
        rotation_angle = map_signal_to_qubit_rotation(avg_val, min_val, max_val)

        # 4. Crie um circuito quântico para cada bloco
        circ = QuantumCircuit(num_qubits, num_qubits)
        circ.ry(rotation_angle, 0)  # Aplica rotação com base no sinal
        circ.measure([0], [0])     # Mede o estado do qubit
        
        # 5. Transpile e execute o circuito no simulador ruidoso
        tcirc = transpile(circ, simulator)
        result = simulator.run(tcirc, shots=1).result() # Uma única medição (colapso da função de onda)
        
        # 6. Converta o resultado da medição em um símbolo
        counts = result.get_counts(circ)
        measured_state = list(counts.keys())[0] # Pega o estado medido ('0' ou '1')
        tape_symbols.append(int(measured_state, 2)) # Converte para 0 ou 1
    
    return tape_symbols

if __name__ == '__main__':
    # ↓ Baixa EDF diretamente do PhysioNet ↓
    url = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"
    local_file = "chb01_01.edf"
    if not os.path.exists(local_file):
        print("Baixando dados EEG...")
        urlretrieve(url, local_file)
        print("Download concluído.")

    # Carrega o arquivo EDF
    raw = mne.io.read_raw_edf(local_file, preload=True, verbose=False)
    data, times = raw[:1] # canal 0
    sinal = data[0]
    sinal_f = bandpass_filter(sinal, fs=raw.info['sfreq'])

    # Use um trecho do sinal para acelerar a simulação
    sinal_sample = sinal_f[:10000] 

    print("Criando fita simbólica quântica...")
    fita_quantica = create_quantum_symbolic_tape(sinal_sample, num_qubits=1, noise_level=0.05)
    print(f"Fita simbólica quântica gerada (primeiros 20 símbolos): {fita_quantica[:20]}")

    # Plot do sinal original e da nova fita
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(times[:len(sinal_sample)], sinal_sample)
    plt.title("EEG Real – Sinal Filtrado")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Voltagem")

    plt.subplot(2, 1, 2)
    # Mapeia os símbolos 0 e 1 para cores para visualização
    cmap = plt.cm.get_cmap('binary', 2)
    plt.imshow([fita_quantica], aspect='auto', cmap=cmap, interpolation='none',
               extent=[0, times[len(sinal_sample) - 1], 0, 1])
    plt.title("Fita Simbólica Quântica")
    plt.xlabel("Tempo (s)")
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    # os.remove(local_file)