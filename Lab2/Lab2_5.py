import numpy as np

def neural_networks(inp, weights):
    prediction = inp * weights
    return prediction

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 150
weights = np.array([0.2, 0.3])
true_prediction = np.array([50,120])
learning_rate = 0.00001

# print(neural_networks(inp,weights))
# print(get_error(true_prediction,neural_networks(inp, weights)))

for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %s, Error: %s, Weight: %s" %(prediction,error,weights) )
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta




