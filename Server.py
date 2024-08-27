import flwr as fl

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
