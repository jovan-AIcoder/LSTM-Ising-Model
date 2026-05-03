import numpy as np
import pandas as pd
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =========================
# USER INPUT
# =========================
def get_user_input():
    print("=== LSTM TRAINING FOR ISING MODEL ===\n")

    size = int(input("Enter lattice size (N): "))
    temperature = float(input("Enter temperature (T): "))
    J = float(input("Enter coupling constant (J): "))
    h = float(input("Enter external magnetic field (h): "))
    steps = int(input("Enter number of steps used in dataset: "))

    return size, temperature, J, h, steps


# =========================
# LOAD DATASET (TRY BLOCK)
# =========================
def load_dataset(size, temperature, J, h, steps):
    filename = (
        f"datasets/Ising_model_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).csv"
    )

    try:
        df = pd.read_csv(filename)
        print(f"\nDataset loaded: {filename} ✨")
        return df
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        return None


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
# MAIN TRAINING
# =========================
def train_lstm():
    while True:
        size, temperature, J, h, steps = get_user_input()
        df = load_dataset(size, temperature, J, h, steps)

        if df is not None:
            break
        print("\nRetry input...\n")

    # =========================
    # SPLIT DATA
    # =========================
    energy = df["Energy"].values.reshape(-1, 1)
    magnetization = df["Magnetization"].values.reshape(-1, 1)

    # =========================
    # NORMALIZATION
    # =========================
    scaler_E = MinMaxScaler()
    scaler_M = MinMaxScaler()

    energy_scaled = scaler_E.fit_transform(energy)
    magnetization_scaled = scaler_M.fit_transform(magnetization)

    # Save scalers
    os.makedirs("scalers", exist_ok=True)

    scaler_E_name = (
        f"scalers/scaler_energy_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).pkl"
    )

    scaler_M_name = (
        f"scalers/scaler_magnetization_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).pkl"
    )

    with open(scaler_E_name, "wb") as f:
        pickle.dump(scaler_E, f)

    with open(scaler_M_name, "wb") as f:
        pickle.dump(scaler_M, f)

    print("Scalers saved ✨")

    # =========================
    # SEQUENCE BUILDING
    # =========================
    window_size = 10

    X_E, y_E = create_sequences(energy_scaled, window_size)
    X_M, y_M = create_sequences(magnetization_scaled, window_size)

    # Combine into multivariate input
    X = np.concatenate((X_E, X_M), axis=2)
    y = np.concatenate((y_E, y_M), axis=1)

    # =========================
    # BUILD MODEL
    # =========================
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window_size, 2)),
        Dense(32, activation="relu"),
        Dense(2)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    print("\nTraining LSTM...\n")

    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)

    model_name = (
        f"models/Ising_LSTM_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).keras"
    )

    model.save(model_name)

    print(f"\nModel saved to: {model_name} 💫")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_lstm()