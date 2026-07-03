import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Part (e): Single Path Simulation ---
T = 1.0
N_e = 10000
dt_e = T / N_e

dW_e = np.random.normal(0, np.sqrt(dt_e), size=N_e)
W_e = np.concatenate([[0], np.cumsum(dW_e)])
increments_e = np.diff(W_e)

QV_e = np.sum(increments_e**2)
TV_e = np.sum(np.abs(increments_e))

print("--- Part (e): Single Path Variation ---")
print(f"Empirical Quadratic Variation (QV): {QV_e:.4f} (Theory: 1.0000)")
print(f"Empirical Total Variation (TV):     {TV_e:.4f} (Theory: Infinite)\n")

# --- Part (f): QV Convergence Plot ---
n_steps_list = [10, 50, 100, 500, 1000, 5000, 10000]
qv_list = []

for N_f in n_steps_list:
    dt_f = T / N_f
    dW_f = np.random.normal(0, np.sqrt(dt_f), size=N_f)
    QV_f = np.sum(dW_f**2)
    qv_list.append(QV_f)

plt.figure(figsize=(8, 4))
plt.semilogx(n_steps_list, qv_list, 'bo-', markersize=6, label='Empirical QV')
plt.axhline(T, color='r', linestyle='--', linewidth=2, label='Theoretical QV = 1.0')
plt.xlabel('Number of partition steps N (log scale)')
plt.ylabel('Quadratic Variation')
plt.title('Convergence of Quadratic Variation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/q1_qv_convergence.png')
plt.show()

# --- Part (g): Marginal Distribution at t=0.5 ---
t_fixed = 0.5
N_g = 1000
dt_g = T / N_g
n_paths = 10000

dW_g = np.random.normal(0, np.sqrt(dt_g), size=(n_paths, N_g))
W_g = np.hstack([np.zeros((n_paths, 1)), np.cumsum(dW_g, axis=1)])
idx = int(t_fixed / dt_g)
W_at_t = W_g[:, idx]

print("--- Part (g): Marginal Distribution at t=0.5 ---")
print(f"Empirical Mean:     {np.mean(W_at_t):.4f} (Theory: 0.0000)")
print(f"Empirical Variance: {np.var(W_at_t):.4f} (Theory: 0.5000)")