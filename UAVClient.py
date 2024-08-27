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
        # Perform training on the device's local data
        self.model.train(self.data['X_train'], self.data['y_train'], epochs=1)
        return self.model.get_weights(), len(self.data['X_train']), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['X_test'], self.data['y_test'])
        return loss, len(self.data['X_test']), {"accuracy": accuracy}
