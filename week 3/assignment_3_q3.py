import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# --- Parameters ---
X0 = 3.0
theta_base = 2.0
sigma = 0.5
T_d = 5.0
N_d = 5000
dt_d = T_d / N_d
n_paths_d = 10

def simulate_ou(theta, T, N, n_paths):
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, N))
    X = np.zeros((n_paths, N + 1))
    X[:, 0] = X0
    for k in range(N):
        X[:, k+1] = X[:, k] - theta * X[:, k] * dt + sigma * dW[:, k]
    return X

# --- Part (d): Simulate 10 Paths ---
X_paths = simulate_ou(theta_base, T_d, N_d, n_paths_d)
t_d = np.linspace(0, T_d, N_d + 1)
stat_std = sigma / np.sqrt(2 * theta_base)

plt.figure(figsize=(10, 5))
for i in range(n_paths_d):
    plt.plot(t_d, X_paths[i], alpha=0.6, lw=1)
plt.axhline(0, color='black', linestyle='--', label='Stationary Mean = 0')
plt.fill_between(t_d, -2*stat_std, 2*stat_std, color='red', alpha=0.15, label='$\pm 2$ Std Dev Band')
plt.title('Ornstein-Uhlenbeck Mean-Reverting Paths')
plt.xlabel('Time (t)')
plt.ylabel('$X_t$')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/q3_ou_paths.png')
plt.show()

# --- Part (e): Verify Stationary Distribution ---
T_e = 10.0
N_e = 10000
n_paths_e = 20000
X_terminal = simulate_ou(theta_base, T_e, N_e, n_paths_e)[:, -1]

theo_stat_var = sigma**2 / (2 * theta_base)
print("--- Part (e): Stationary Distribution ---")
print(f"Empirical Mean: {np.mean(X_terminal):.4f} (Theory: 0.0000)")
print(f"Empirical Var:  {np.var(X_terminal):.4f} (Theory: {theo_stat_var:.4f})\n")

plt.figure(figsize=(8, 4))
plt.hist(X_terminal, bins=70, density=True, alpha=0.6, color='green', label='Simulated $X_{10}$')
x_axis = np.linspace(X_terminal.min(), X_terminal.max(), 300)
plt.plot(x_axis, norm.pdf(x_axis, 0, np.sqrt(theo_stat_var)), 'r-', lw=2, label='Theoretical Stationary Density')
plt.title('Stationary Distribution of OU Process')
plt.xlabel('$X_T$')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/q3_ou_stationary.png')
plt.show()

# --- Part (f): Effect of Theta ---
thetas = [0.5, 2.0, 5.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, th in enumerate(thetas):
    X_th = simulate_ou(th, T_d, N_d, 3) # Plot 3 paths for clarity
    for j in range(3):
        axes[i].plot(t_d, X_th[j], alpha=0.7)
    axes[i].axhline(0, color='black', linestyle='--')
    axes[i].set_title(f'Theta = {th}')
    axes[i].set_xlabel('Time (t)')

axes[0].set_ylabel('$X_t$')
plt.tight_layout()
plt.savefig('../outputs/q3_ou_thetas.png')
plt.show()