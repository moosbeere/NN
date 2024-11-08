import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deliv(x):
    return x*(1-x)

inp = np.array([[0,0],[0,1],[1,0],[1,1]])
true_prediction = np.array([[0],[1],[1],[0]])

input_size = len(inp[0])
hid_size = 10
output_size = len(true_prediction[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hid_size))
weight_out = np.random.uniform(size=(hid_size, output_size))

learning_rate = 0.1
epochs = 100000

for i in range(epochs):
    layer_hid = sigmoid(inp.dot(weight_hid))
    # print(layer_hid)
    # exit(0)
    layer_out = sigmoid(layer_hid.dot(weight_out))
    error = (layer_out - true_prediction) ** 2
    # print(layer_out)
    # exit(0)
    out_delta = (layer_out - true_prediction) * sigmoid_deliv(layer_out)
    hid_delta = out_delta.dot(weight_out.T) * sigmoid_deliv(layer_hid)
    weight_out -= learning_rate * layer_hid.T.dot(out_delta)
    weight_hid -= learning_rate * inp.T.dot(hid_delta)

    if (i % 1000) == 0:
        error = np.mean(error)
        print(f"Epoch {i}, Error {error}")

inp_test = np.array([[1,1]])
layer_hid = sigmoid(inp_test.dot(weight_hid))
layer_out = sigmoid(layer_hid.dot(weight_out))
print(f"Accuracy: {layer_out}")