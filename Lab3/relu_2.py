import numpy as np

def relu(x):
    return (x > 0) * x

def relu_deliv(x):
    return (x > 0)

inp = np.array([[0,0],[0,1],[1,0],[1,1]])
true_prediction = np.array([[0],[1],[1],[0]])

input_size = len(inp[0])
hid_size = 4
output_size = len(true_prediction[0])

np.random.seed(100)
weight_hid = 2 * np.random.random((input_size, hid_size)) - 1
# print(weight_hid)
weight_out = 2 * np.random.random((hid_size, output_size)) - 1
# print(weight_out)

learning_rate = 0.1
epochs = 5000

for i in range(epochs):
    layer_hid = relu(inp.dot(weight_hid))
    layer_out = layer_hid.dot(weight_out)
    error = (layer_out - true_prediction) ** 2
    out_delta = (layer_out - true_prediction)
    hid_delta = out_delta.dot(weight_out.T) * relu_deliv(layer_hid)
    weight_out -= learning_rate * layer_hid.T.dot(out_delta)
    weight_hid -= learning_rate * inp.T.dot(hid_delta)

    if (i % 1000) == 0:
        error = np.mean(error)
        print(f"Epoch {i}, Error {error}")

inp_test = np.array([[0,1]])
layer_hid = relu(inp_test.dot(weight_hid))
layer_out = relu(layer_hid.dot(weight_out))
print(f"Prediction: {layer_out}")