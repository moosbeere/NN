def neural_networks(inp, weights):
    prediction = [0] * len(weights)
    for i in range(len(weights)):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction[i] = ws
    return prediction

weight_1 = [0.5, 0.3, 0.6]
weight_2 = [0.2, 0.3, 0.7]
weight_3 = [0.2, 0.3, 0.7]

weights = [weight_1, weight_2, weight_3]
inp = [20,7,30]

print(neural_networks(inp, weights))



