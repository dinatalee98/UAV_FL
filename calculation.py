import numpy as np
import cvxpy as cp

###
### distance
###

def euclidean_distance(uav_pos, device_pos, altitude):
    return np.sqrt(np.sum((uav_pos - device_pos) ** 2) + altitude ** 2)

# Example usage:
uav_pos = np.array([x_u, y_u])  # UAV position at round n
device_pos = np.array([x_k, y_k])  # IoT device position
altitude = H_u  # UAV altitude
distance = euclidean_distance(uav_pos, device_pos, altitude)

###
### Computation model
###

def computation_time_energy(c_k, D_k, f_k, a_k, alpha_k):
    t_comp = (c_k * D_k) / f_k
    E_comp = a_k * (alpha_k / 2) * c_k * D_k * (f_k ** 2)
    return t_comp, E_comp

# Example usage:
t_comp, E_comp = computation_time_energy(c_k, D_k, f_k, a_k, alpha_k)

###
### Communication model
###

def path_loss(d_k, fc, eta_LoS, eta_NLoS, P_LoS):
    PL_LoS = 20 * np.log10((4 * np.pi * fc * d_k) / c) + eta_LoS
    PL_NLoS = 20 * np.log10((4 * np.pi * fc * d_k) / c) + eta_NLoS
    return P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

# Example usage:
PL = path_loss(d_k, fc, eta_LoS, eta_NLoS, P_LoS)

def data_rate_and_energy(b_k, p_k, h_k, sigma2, a_k, s):
    R_k = b_k * np.log2(1 + (p_k * np.linalg.norm(h_k) ** 2) / sigma2)
    t_comm = (a_k * s) / R_k
    E_comm = (t_comm * sigma2 * (2 ** (a_k * s / (b_k * t_comm)) - 1)) / np.linalg.norm(h_k) ** 2
    return R_k, t_comm, E_comm

# Example usage:
R_k, t_comm, E_comm = data_rate_and_energy(b_k, p_k, h_k, sigma2, a_k, s)

###
### Optimization
###

# Variables
a_k = cp.Variable((K, N), boolean=True)  # Scheduling variable for K devices and N rounds

# Objective Function
objective = cp.Minimize(2 / (N * eta) * (F_w0 - F_star) + ... )  # Continue with the formula

# Constraints
constraints = [
    cp.sum(b_k) <= B,  # Bandwidth constraint
    cp.sum(E_comm + E_comp) <= E_k,  # Energy constraint
    t_comm <= epsilon  # Time constraint
]

# Problem Definition
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

print("Optimal Scheduling Variables:", a_k.value)