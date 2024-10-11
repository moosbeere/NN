import numpy as np

def neural_networks(inp, weights):
    prediction_h = inp.dot(weights[0])
    prediction_out = prediction_h.dot(weights[1])
    return prediction_out

weight_h = np.array([[0.3, 0.4], [0.5, 0.7]]).T
weight_out = np.array([[0.2, 0.6], [0.3, 0.8]]).T
weights = [weight_h, weight_out]
inp = np.array([40, 50])

print(neural_networks(inp, weights))



