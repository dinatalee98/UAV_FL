import numpy as np
import cvxpy as cp

c = 3.8

def euclidean_distance(uav_pos, device_pos, altitude): # System model d_k(n)
    return np.sqrt(np.sum((uav_pos - device_pos) ** 2) - altitude ** 2)

def computation_time_energy(c_k, D_k, f_k, a_k, alpha_k): # Eq 3 & 4
    t_comp = (c_k * D_k) / f_k
    E_comp = a_k * (alpha_k / 2) * c_k * D_k * (f_k ** 2)
    return t_comp, E_comp

def P_LoS(n, k, H_u, d):
    zeta1 = 1
    zeta2 = 2
    theta_k = np.arctan(H_u/d[k][n]) # d[k][n] distance between UAV and IoT
    return 1 / (1 + zeta1 * np.exp(-zeta2 * (180 / np.pi * theta_k - zeta1)))

def PL_LoS(k, n, f_c, d, eta_LoS):
    return 20 * np.log10((4 * np.pi * f_c * d[k][n]) / c) + eta_LoS

def PL_NLoS(k, n, f_c, d, eta_NLoS):
    return 20 * np.log10((4 * np.pi * f_c * d[k][n]) / c) + eta_NLoS

def path_loss(k, n, d, f_c, eta_LoS, eta_NLoS, P_LoS): # Eq 6 & 7 & 8
    return P_LoS * PL_LoS(k, n, f_c, d, eta_LoS) + (1 - P_LoS) * PL_NLoS(k, n, f_c, d, eta_NLoS)


def channel_coefficient(alpha_0, d, beta, PL_LoS, PL_NLoS): # Eq 9, 10, 11
    large_scale_fading = alpha_0 * d ** beta
    small_scale_fading = np.sqrt((K / (K+1)) * PL_LoS) + np.sqrt((1 / (K+1)) * PL_NLoS)
    h_k = np.sqrt(large_scale_fading * small_scale_fading)
    return h_k


def data_rate_and_energy(n, b_k, p_k, h_k, sigma, a_k, s): # Eq 12, 13, 14
    R_k = b_k[n] * np.log2(1 + (p_k[n] * np.linalg.norm(h_k[n]) ** 2) / sigma ** 2)
    t_comm = (a_k[n] * s) / R_k
    E_comm = (t_comm * (sigma ** 2) * (2 ** (a_k[n] * s / (b_k[n] * t_comm)) - 1)) / np.linalg.norm(h_k[n]) ** 2
    return R_k, t_comm, E_comm



####################

#Optimize part

####################

K = 40 # number of IoT devices
N = 30 # number of round

# Variables
a_k = cp.Variable((K, N), boolean=True)  # Scheduling variable for K devices and N rounds

# Objective Function
P1_1 = 2 / (N * eta) * (F_w0 - F_star) #18번식 앞 부분

P1_2 = 0
for n in range(N):
    for k in range(K):
        P1_2 += (1-a_k[k][n]) * (D_k ** 2)

P1_2 = P1_2 * 4 * K * kappa / (N * (D ** 2)) # 18번 식

objective = cp.Minimize(P1_1 + P1_2)  # Continue with the formula

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

