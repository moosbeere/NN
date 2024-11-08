import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deliv(x):
    return 1 - x**2

inp = np.array([[0,0],[0,1],[1,0],[1,1]])
true_prediction = np.array([[0],[1],[1],[0]])

input_size = len(inp[0])
hid_size = 4
output_size = len(true_prediction[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hid_size))
weight_out = np.random.uniform(size=(hid_size, output_size))

learning_rate = 0.1
epochs = 3000

for i in range(epochs):
    layer_hid = tanh(inp.dot(weight_hid))
    # print(layer_hid)
    # exit(0)
    layer_out = tanh(layer_hid.dot(weight_out))
    error = (layer_out - true_prediction) ** 2
    # print(layer_out)
    # exit(0)
    out_delta = (layer_out - true_prediction) * tanh_deliv(layer_out)
    hid_delta = out_delta.dot(weight_out.T) * tanh_deliv(layer_hid)
    weight_out -= learning_rate * layer_hid.T.dot(out_delta)
    weight_hid -= learning_rate * inp.T.dot(hid_delta)

    if (i % 1000) == 0:
        error = np.mean(error)
        print(f"Epoch {i}, Error {error}")

inp_test = np.array([[1,1]])
layer_hid = tanh(inp_test.dot(weight_hid))
layer_out = tanh(layer_hid.dot(weight_out))
print(f"Accuracy: {layer_out}")