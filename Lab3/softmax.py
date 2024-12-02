import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deliv(x):
    return x*(1-x)

def softmax(x):
    exp = np.exp(x)
    # print(f"exp: {exp}")
    return exp/np.sum(exp, axis=1, keepdims=True)

# temp = np.array([[1,2,3], [4,5,6]])
# print(np.sum(temp, axis=0, keepdims=True))
# exit(0)

x = (np.array
    ([
         [0, 0, 0, 0], # 0
         [0, 0, 0, 1], # 1
         [0, 0, 1, 0], # 2
         [0, 0, 1, 1], # 3
         [0, 1, 0, 0], # 4
         [0, 1, 0, 1], # 5
         [0, 1, 1, 0], # 6
         [0, 1, 1, 1], # 7
         [1, 0, 0, 0], # 8
         [1, 0, 0, 1], # 9
         [1, 0, 1, 0], # 10
         [1, 0, 1, 1], # 11
         [1, 1, 0, 0], # 12
         [1, 1, 0, 1], # 13
         [1, 1, 1, 0], # 14
         [1, 1, 1, 1], # 15
]))

y = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

])

input_size = len(x[0])
hid_size = 15
output_size = len(y[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hid_size))
weight_out = np.random.uniform(size=(hid_size, output_size))

learning_rate = 0.1
epochs = 10000

for i in range(epochs):
    layer_hid = sigmoid(x.dot(weight_hid))
    # print(layer_hid)
    # exit(0)
    layer_out = softmax(layer_hid.dot(weight_out))
    error = (layer_out - y) ** 2
    # print(layer_out)
    # exit(0)
    out_delta = (layer_out - y)/len(layer_out)
    hid_delta = out_delta.dot(weight_out.T) * sigmoid_deliv(layer_hid)
    weight_out -= learning_rate * layer_hid.T.dot(out_delta)
    weight_hid -= learning_rate * x.T.dot(hid_delta)

    if (i % 1000) == 0:
        error = np.mean(error)
        print(f"Epoch {i}, Error {error}")

def predict(inp):
    layer_hid = sigmoid(inp.dot(weight_hid))
    layer_out = softmax(layer_hid.dot(weight_out))
    print(layer_out)
    return np.argmax(layer_out)

for inp in x:
    print("----------------")
    # print(np.array(inp))
    print(f"Предсказанное значениe для {inp}", predict(np.array([inp])))