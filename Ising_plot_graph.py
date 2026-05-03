import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# =========================
# USER INPUT
# =========================
def get_user_input():
    print("=== ISING MODEL PREDICTION VISUALIZATION ===\n")

    size = int(input("Enter lattice size (N): "))
    temperature = float(input("Enter temperature (T): "))
    J = float(input("Enter coupling constant (J): "))
    h = float(input("Enter external magnetic field (h): "))
    steps = int(input("Enter number of steps: "))

    return size, temperature, J, h, steps


# =========================
# LOAD DATA
# =========================
def load_dataset(size, temperature, J, h, steps):
    filename = (
        f"datasets/Ising_model_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).csv"
    )

    try:
        df = pd.read_csv(filename)
        print(f"Dataset loaded: {filename}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


# =========================
# LOAD MODEL + SCALERS
# =========================
def load_model_and_scalers(size, temperature, J, h, steps):
    model_name = (
        f"models/Ising_LSTM_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).keras"
    )

    scaler_E_name = (
        f"scalers/scaler_energy_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).pkl"
    )

    scaler_M_name = (
        f"scalers/scaler_magnetization_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).pkl"
    )

    model = load_model(model_name)

    with open(scaler_E_name, "rb") as f:
        scaler_E = pickle.load(f)

    with open(scaler_M_name, "rb") as f:
        scaler_M = pickle.load(f)

    print("Model and scalers loaded ✨")

    return model, scaler_E, scaler_M


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


# =========================
# MAIN
# =========================
def main():
    while True:
        size, temperature, J, h, steps = get_user_input()
        df = load_dataset(size, temperature, J, h, steps)

        if df is not None:
            break
        print("Retry input...\n")

    model, scaler_E, scaler_M = load_model_and_scalers(
        size, temperature, J, h, steps
    )

    # =========================
    # PREPARE DATA
    # =========================
    energy = df["Energy"].values.reshape(-1, 1)
    magnetization = df["Magnetization"].values.reshape(-1, 1)

    energy_scaled = scaler_E.transform(energy)
    magnetization_scaled = scaler_M.transform(magnetization)

    window_size = 10

    X_E, y_E = create_sequences(energy_scaled, window_size)
    X_M, y_M = create_sequences(magnetization_scaled, window_size)

    X = np.concatenate((X_E, X_M), axis=2)

    # =========================
    # PREDICTION
    # =========================
    predictions = model.predict(X)

    pred_E = scaler_E.inverse_transform(predictions[:, 0].reshape(-1, 1))
    pred_M = scaler_M.inverse_transform(predictions[:, 1].reshape(-1, 1))

    # ground truth (aligned)
    true_E = energy[window_size:]
    true_M = magnetization[window_size:]

    # =========================
    # PLOTTING
    # =========================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Energy plot
    axes[0].plot(true_E, label="True Energy", color="black")
    axes[0].plot(pred_E, label="Predicted Energy", linestyle="--")
    axes[0].set_title("Energy vs Time")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    # Magnetization plot
    axes[1].plot(true_M, label="True Magnetization", color="blue")
    axes[1].plot(pred_M, label="Predicted Magnetization", linestyle="--")
    axes[1].set_title("Magnetization vs Time")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Magnetization")
    axes[1].legend()

    plt.tight_layout()

    # =========================
    # SAVE GRAPH
    # =========================
    os.makedirs("graphs", exist_ok=True)

    filename = (
        f"graphs/Ising_graph_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).png"
    )

    plt.savefig(filename)

    print(f"\nGraph saved to: {filename} ✨")

    plt.show()


if __name__ == "__main__":
    main()