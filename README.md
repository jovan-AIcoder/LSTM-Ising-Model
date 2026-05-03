# LSTM for Time-Series Ising Model

## 📌 Overview

This project explores whether a neural network can learn and predict the stochastic dynamics of the 2D Ising model using only macroscopic observables: **Energy** and **Magnetization**.

Instead of explicitly using the Hamiltonian or transition probabilities, we train a Long Short-Term Memory (LSTM) network to model the temporal evolution of the system.

> Can a neural network learn statistical physics purely from observation?

---

## 🧠 Core Idea

The Ising model evolves as a stochastic process governed by local spin interactions. While the underlying dynamics occur at the microscopic level (spin configurations), we restrict the model to observe only:

* Energy (E(t))
* Magnetization (M(t))

This creates a **coarse-grained time series**, and the LSTM attempts to predict:

$[
(E, M)*t \rightarrow (E, M)*{t+1}
]$

---

## ⚙️ Project Structure

```
.
├── datasets/     # Generated CSV datasets
├── scalers/      # Saved normalization scalers (.pkl)
├── models/       # Trained LSTM models
├── graphs/       # Prediction vs ground truth plots
│
├── Ising_datagen.py
├── Ising_train_model.py
├── Ising_plot_graph.py
└── README.md
```

---

## 🔬 Pipeline

### 1. Dataset Generation

Simulate the 2D Ising model using the Metropolis algorithm and export:

* Energy
* Magnetization

Saved as:

```
datasets/Ising_model_(size=...,temperature=...,J=...,h=...,steps=...).csv
```

---

### 2. LSTM Training

* Input: sequences of Energy and Magnetization
* Output: next-step prediction

Features:

* MinMax normalization
* Sliding window sequence construction
* Multivariate LSTM

Artifacts:

* Model → `models/`
* Scalers → `scalers/`

---

### 3. Visualization

Compare:

* Ground truth vs predicted values

Outputs:

* Energy vs Time
* Magnetization vs Time

Saved as:

```
graphs/Ising_graph_(...).png
```

---

## 🚀 Usage

### Step 1 — Generate Dataset

```bash
python Ising_datagen.py
```

---

### Step 2 — Train LSTM

```bash
python Ising_train_model.py
```

---

### Step 3 — Visualize Results

```bash
python Ising_plot_graph.py
```

---

## 📊 Example Output

The model produces plots comparing predicted and true trajectories of:

* Energy
* Magnetization

These reveal how well the LSTM captures the stochastic dynamics.

---

## 🌡️ Scientific Insight

This project implicitly investigates:

* Information loss under coarse-graining
* Temporal correlations in statistical systems
* Whether macroscopic observables are sufficient for prediction

Possible outcomes:

* ✔️ Good prediction → strong temporal structure in observables
* ❌ Poor prediction → hidden dependence on microscopic states

---

## 🔮 Future Work

* Multi-step forecasting
* Critical temperature detection
* Comparison across different lattice sizes
* Incorporating full spin configurations (ConvLSTM)
* Transfer learning to real-world stochastic systems

---

## 📚 Requirements

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

---

## 💬 Final Thought

This project is not just about training a neural network.

It is an experiment:

> Can a machine rediscover the behavior of a physical system without ever being told its laws?

---

## 👤 Author

Jovan
Physics & AI Enthusiast
Licensed under MIT License

---
