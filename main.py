if __name__ == "__main__":
    # Start the server
    start_server()

    # Start clients
    for i in range(num_devices):
        run_client(device_id=i, data=get_device_data(i))
