def neural_networks(inp, weights):
    prediction = [0] * len(weights)
    for i in range(len(weights[0])):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weights[0][i][j]
        prediction[i] = ws
    print(prediction)
    for i in range(len(weights[1])):
        ws = 0
        for j in range(len(prediction)):
            ws += prediction[j] * weights[1][i][j]
        prediction[i] = ws
    return prediction

weight_h = [[0.3, 0.4], [0.5, 0.7], [0.4, 0.3], [0.6, 0.7], [0.6,0.4]]
weight_out = [[0.2, 0.6], [0.3, 0.8]]
weights = [weight_h, weight_out]
inp = [40, 50]


print(neural_networks(inp, weights))



