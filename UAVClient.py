class UAVClient(fl.client.NumPyClient):
    # ...

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
