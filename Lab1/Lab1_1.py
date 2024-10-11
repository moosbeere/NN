def neural_networks(inp, weight):
    prediction = inp * weight
    return prediction

inp = 43
weight = 0.2

out_1 = neural_networks(inp, weight)
print(out_1)