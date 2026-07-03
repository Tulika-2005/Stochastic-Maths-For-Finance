import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

np.random.seed(42)

# --- Parameters ---
S0 = 100.0
mu = 0.07
sigma_base = 0.2
T = 1.0
N = 500
n_paths = 50000
dt = T / N

def simulate_gbm(sigma):
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, N))
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    for k in range(N):
        S[:, k+1] = S[:, k] + mu * S[:, k] * dt + sigma * S[:, k] * dW[:, k]
    return S[:, -1]

# --- Part (d) & (e): Simulation and Log-Normal Plot ---
ST_base = simulate_gbm(sigma_base)

theoretical_mean = S0 * np.exp(mu * T)
theoretical_std = S0 * np.exp(mu * T) * np.sqrt(np.exp(sigma_base**2 * T) - 1)

print("--- Part (e): Euler-Maruyama Terminal Price ---")
print(f"Empirical Mean: {np.mean(ST_base):.4f} (Theory: {theoretical_mean:.4f})")
print(f"Empirical Std:  {np.std(ST_base):.4f} (Theory: {theoretical_std:.4f})\n")

plt.figure(figsize=(8, 4))
plt.hist(ST_base, bins=100, density=True, alpha=0.6, color='steelblue', label='Simulated $S_T$')

# Overlay theoretical log-normal density
log_mean = np.log(S0) + (mu - 0.5 * sigma_base**2) * T
log_std = sigma_base * np.sqrt(T)
x = np.linspace(ST_base.min(), ST_base.max(), 300)
plt.plot(x, lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)), 'r-', lw=2, label='Theoretical Log-Normal')
plt.title('Terminal Stock Price Distribution ($S_T$)')
plt.xlabel('$S_T$')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/q2_gbm_histogram.png')
plt.show()

# --- Part (f): Volatility Impact ---
print("--- Part (f): Volatility Impact ---")
K = 110.0
sigmas = [0.1, 0.3, 0.5]
for sig in sigmas:
    ST_sig = simulate_gbm(sig)
    payoff = np.maximum(ST_sig - K, 0)
    print(f"Sigma = {sig}: Mean = {np.mean(ST_sig):.2f}, Std = {np.std(ST_sig):.2f}, Expected Payoff = {np.mean(payoff):.2f}")
print()

# --- Part (g): Log Returns ---
log_returns = np.log(ST_base / S0)
theo_log_mean = (mu - 0.5 * sigma_base**2) * T
theo_log_var = sigma_base**2 * T

print("--- Part (g): Log-Returns ---")
print(f"Empirical Log-Return Mean: {np.mean(log_returns):.4f} (Theory: {theo_log_mean:.4f})")
print(f"Empirical Log-Return Var:  {np.var(log_returns):.4f} (Theory: {theo_log_var:.4f})")