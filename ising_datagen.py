import numpy as np
import pandas as pd
import os
import sys


def initialize_lattice(size):
    rng = np.random.default_rng()
    return rng.choice([-1, 1], size=(size, size))


def neighbor_sum(spins, i, j):
    N = spins.shape[0]
    return (
        spins[(i + 1) % N, j]
        + spins[(i - 1) % N, j]
        + spins[i, (j + 1) % N]
        + spins[i, (j - 1) % N]
    )


def delta_energy(spins, i, j, J, h):
    s = spins[i, j]
    return 2 * s * (J * neighbor_sum(spins, i, j) + h)


def metropolis_step(spins, beta, J, h, rng):
    N = spins.shape[0]
    for _ in range(N * N):
        i = rng.integers(N)
        j = rng.integers(N)
        dE = delta_energy(spins, i, j, J, h)
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            spins[i, j] = -spins[i, j]


def total_energy(spins, J, h):
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(N):
            energy -= J * spins[i, j] * (
                spins[(i + 1) % N, j] + spins[i, (j + 1) % N]
            )
            energy -= h * spins[i, j]
    return energy


def total_magnetization(spins):
    return spins.sum()


# 🌿 Progress Bar Function
def print_progress(step, total, bar_length=30):
    progress = step / total
    filled = int(bar_length * progress)
    bar = "-" * filled + " " * (bar_length - filled)

    # ANSI green color
    GREEN = "\033[92m"
    RESET = "\033[0m"

    sys.stdout.write(
        f"\r{GREEN}{step}/{total} [{bar}]{RESET}"
    )
    sys.stdout.flush()


def run_simulation(size, temperature, J, h, steps):
    beta = 1.0 / temperature
    rng = np.random.default_rng()
    spins = initialize_lattice(size)

    energy_history = []
    mag_history = []

    for step in range(steps):
        metropolis_step(spins, beta, J, h, rng)

        energy_history.append(total_energy(spins, J, h))
        mag_history.append(total_magnetization(spins))

        # update progress bar
        print_progress(step + 1, steps)

    print()  # newline after progress bar

    return energy_history, mag_history


def main():
    print("=== ISING MODEL CSV DATA GENERATOR ===\n")

    size = int(input("Enter lattice size (N): "))
    temperature = float(input("Enter temperature (T): "))
    J = float(input("Enter coupling constant (J): "))
    h = float(input("Enter external magnetic field (h): "))
    steps = int(input("Enter number of simulation steps: "))

    print("\nRunning simulation...\n")

    energy, magnetization = run_simulation(
        size=size,
        temperature=temperature,
        J=J,
        h=h,
        steps=steps,
    )

    df = pd.DataFrame({
        "Energy": energy,
        "Magnetization": magnetization
    })

    os.makedirs("datasets", exist_ok=True)

    filename = (
        f"datasets/Ising_model_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).csv"
    )

    df.to_csv(filename, index=False)

    print(f"\nDataset saved to: {filename} ✨")


if __name__ == "__main__":
    main()