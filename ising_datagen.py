import numpy as np
import pandas as pd
import os
import sys
from numba import njit  

# 1. Initialize the lattice using Numba-compatible random methods
@njit
def initialize_lattice(size):
    """
    Initializes a 2D lattice grid with random spins of 1 or -1.
    Uses np.random.rand for Numba compatibility.
    """
    return np.where(np.random.rand(size, size) < 0.5, -1, 1).astype(np.float64)

@njit
def neighbor_sum(spins, i, j):
    """
    Calculates the sum of the 4 nearest neighbors with Periodic Boundary Conditions (PBC).
    """
    N = spins.shape[0]
    return (
        spins[(i + 1) % N, j]
        + spins[(i - 1) % N, j]
        + spins[i, (j + 1) % N]
        + spins[i, (j - 1) % N]
    )

@njit
def calculate_initial_energy(spins, J, h):
    """
    Computes the total baseline energy of the initial lattice state from scratch.
    Time complexity: O(N^2) - only executed once at the start.
    """
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(N):
            energy -= J * spins[i, j] * (
                spins[(i + 1) % N, j] + spins[i, (j + 1) % N]
            )
            energy -= h * spins[i, j]
    return energy

@njit
def run_metropolis_and_track(spins, steps, beta, J, h):
    """
    Executes the core Metropolis simulation loop and logs thermodynamic properties.
    Everything inside is compiled to machine code via Numba for extreme performance.
    """
    N = spins.shape[0]
    energy_history = np.zeros(steps)
    mag_history = np.zeros(steps)
    
    # Calculate starting values
    current_energy = calculate_initial_energy(spins, J, h)
    current_mag = np.sum(spins)
    
    for step in range(steps):
        # 1 Monte Carlo Step (MCS) consists of N*N spin-flip attempts
        for _ in range(N * N):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            
            s = spins[i, j]
            dE = 2 * s * (J * neighbor_sum(spins, i, j) + h)
            
            # Metropolis acceptance criterion
            if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
                spins[i, j] = -s
                # Dynamic O(1) updates instead of recalculating the entire lattice grid
                current_energy += dE
                current_mag += 2 * (-s)
                
        energy_history[step] = current_energy
        mag_history[step] = current_mag
        
    return energy_history, mag_history


# Progress Bar Function (Kept in pure Python due to I/O streaming)
def print_progress(step, total, bar_length=30):
    """
    Displays an animated progress bar in the terminal window.
    """
    progress = step / total
    filled = int(bar_length * progress)
    bar = "-" * filled + " " * (bar_length - filled)
    GREEN = "\033[92m"
    RESET = "\033[0m"
    sys.stdout.write(f"\r{GREEN}{step}/{total} [{bar}]{RESET}")
    sys.stdout.flush()


def main():
    print("=== ISING MODEL CSV DATA GENERATOR (TURBO VERSION) ===\n")

    # Interactive User Configurations
    size = int(input("Enter lattice size (N): "))
    temperature = float(input("Enter temperature (T): "))
    J = float(input("Enter coupling constant (J): "))
    h = float(input("Enter external magnetic field (h): "))
    steps = int(input("Enter number of simulation steps: "))

    print("\nCompiling and running simulation (Numba takes ~1-2 sec for the first run)...\n")

    beta = 1.0 / temperature
    spins = initialize_lattice(size)

    # Triggering the highly optimized simulation loop
    energy, magnetization = run_metropolis_and_track(spins, steps, beta, J, h)

    # Flush final progress status bar
    print_progress(steps, steps)
    print() 

    # Compile simulation array outputs into a structured DataFrame
    df = pd.DataFrame({
        "Energy": energy,
        "Magnetization": magnetization
    })

    # Export datasets to disk
    os.makedirs("datasets", exist_ok=True)
    filename = (
        f"datasets/Ising_model_"
        f"(size={size},temperature={temperature},J={J},h={h},steps={steps}).csv"
    )

    df.to_csv(filename, index=False)
    print(f"\nDataset successfully saved to: {filename} ✨")


if __name__ == "__main__":
    main()