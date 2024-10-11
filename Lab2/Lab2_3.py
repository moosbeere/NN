import numpy as np

def neural_networks(inp, weights):
    prediction = inp * weights
    return prediction

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 20
weights = 0.3
true_prediction = 70
learning_rate = 0.001

# print(neural_networks(inp,weights))
# print(get_error(true_prediction,neural_networks(inp, weights)))

for i in range(55):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Error: %.20f, Weight: %.10f" %(prediction,error,weights) )
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta




