def neural_networks(inp, weights):
    prediction = [0] * len(weights)
    for i in range(len(weights)):
        prediction[i] = inp * weights[i]
    return prediction

print(neural_networks(23, [0.5, 0.6, 0.1]))



