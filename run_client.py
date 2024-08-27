def run_client(device_id, data):
    model = create_model()
    client = UAVClient(device_id, data, model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

# Example: Running the first client
data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
run_client(device_id=1, data=data)
