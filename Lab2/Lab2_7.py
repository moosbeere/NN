import numpy as np

def neural_networks(inp, weights):
    prediction = inp.dot(weights)
    return prediction

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([
    [150,40],
    [175,70],
    [180,90]
])
weights = np.array([0.2, 0.3])
true_prediction = np.array([0, 100, 100])
learning_rate = 0.001

# print(neural_networks(inp,weights))
# print(get_error(true_prediction,neural_networks(inp, weights)))

for i in range(100):
    error = 0
    delta = 0
    for j in range(len(inp)):
        current_inp = inp[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction[j], prediction)
        print("Prediction: %s, True_prediction: %s, Weight: %s" %(prediction,true_prediction[j],weights) )
        delta += (prediction - true_prediction[j]) * current_inp * learning_rate
    weights = weights - delta/len(inp)
    print("------------------------------------------")
    print("Errors: %s" %(error))



