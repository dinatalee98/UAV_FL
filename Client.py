import flwr as fl
import numpy as np

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
