import tensorflow as tf
from keras import layers, models
import flwr as fl
import numpy as np
import cvxpy as cp

class Client(fl.client.NumPyClient):
    def __init__(self, device_id, data, model):
        self.device_id = device_id
        self.data = data
        self.model = model

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Calculate energy constraints
        remaining_energy = calculate_remaining_energy(self.device_id)
        if remaining_energy < threshold:
            # Reduce training time or skip a round
            epochs = 1
        else:
            epochs = 5

        self.model.train(self.data['X_train'], self.data['y_train'], epochs=epochs)
        return self.model.get_weights(), len(self.data['X_train']), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['X_test'], self.data['y_test'])
        return loss, len(self.data['X_test']), {"accuracy": accuracy}

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_client(device_id, data):
    model = create_model()
    client = Client(device_id, data, model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

# Define the aggregation function
def weighted_average(metrics):
    accuracies = [num_examples * accuracy for num_examples, accuracy in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Start the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config={"num_rounds": 10},
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
)

def euclidean_distance(uav_pos, device_pos, altitude):
    return np.sqrt(np.sum((uav_pos - device_pos) ** 2) + altitude ** 2)

def computation_time_energy(c_k, D_k, f_k, a_k, alpha_k):
    t_comp = (c_k * D_k) / f_k
    E_comp = a_k * (alpha_k / 2) * c_k * D_k * (f_k ** 2)
    return t_comp, E_comp

def LoS_prob(H_u, d_k):
    theta_k = np.arctan(H_u / d_k)
    zeta1 = 3 # constants depending on theta_k s
    zeta2 = 3
    return(1 / (1 + zeta1 * np.exp(-zeta2(180/np.pi)*theta_k - zeta1)))

def path_loss(d_k, f_c, eta_LoS, eta_NLoS, P_LoS):
    c = 300000 #빛의 속도
    PL_LoS = 20 * np.log10((4 * np.pi * f_c * d_k) / c) + eta_LoS
    PL_NLoS = 20 * np.log10((4 * np.pi * f_c * d_k) / c) + eta_NLoS
    return P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

def data_rate_and_energy(b_k, p_k, h_k, sigma2, a_k, s):
    R_k = b_k * np.log2(1 + (p_k * np.linalg.norm(h_k) ** 2) / sigma2)
    t_comm = (a_k * s) / R_k
    E_comm = (t_comm * sigma2 * (2 ** (a_k * s / (b_k * t_comm)) - 1)) / np.linalg.norm(h_k) ** 2
    return R_k, t_comm, E_comm

if __name__ == "__main__":
    uav_pos = np.array([x_u, y_u])  # UAV position at round n
    device_pos = np.array([x_k, y_k])  # IoT device position
    altitude = H_u  # UAV altitude
    distance = euclidean_distance(uav_pos, device_pos, altitude)

    # computation
    t_comp, E_comp = computation_time_energy(c_k, D_k, f_k, a_k, alpha_k)

    # Start the server
    start_server()
    # Example: Running the first client
    data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    run_client(device_id=1, data=data)

    # Start clients
    for i in range(num_devices):
        run_client(device_id=i, data=get_device_data(i))

    ###
    ### Optimization
    ###

    # Variables
    a_k = cp.Variable((K, N), boolean=True)  # Scheduling variable for K devices and N rounds

    # Objective Function
    objective = cp.Minimize(2 / (N * eta) * (F_w0 - F_star) + 4 * K * kappa / N * (D ** 2) )  # Continue with the formula

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