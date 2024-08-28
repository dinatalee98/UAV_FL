import tensorflow as tf
from keras import layers, models
import flwr as fl

class UAVClient(fl.client.NumPyClient):
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

def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_client(device_id, data):
    model = create_model()
    client = UAVClient(device_id, data, model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

# Example: Running the first client
data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
run_client(device_id=1, data=data)

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



if __name__ == "__main__":
    # Start the server
    start_server()

    # Start clients
    for i in range(num_devices):
        run_client(device_id=i, data=get_device_data(i))
