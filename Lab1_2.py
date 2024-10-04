def neural_networks(inp, weights):
    prediction = 0
    for i in range(len(weights)):
        prediction += inp[i] * weights[i]
    return prediction

print(neural_networks([23,40], [0.1, 0.5]))
