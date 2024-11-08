import numpy as np

def neural_networks(inp, weights):
    prediction = inp.dot(weights)
    return prediction

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def relu(x):
    return (x > 0) * x

def reluderiv(x):
    return (x > 0)

inp = np.array([
    [150,40],
    [175,70],
    [180,90],
    [200,20]
])
true_prediction = np.array([[10, 20, 25, 15]]).T

layer_hid_size = 3
layer_inp_size = len(inp[0])
layer_out_size = len(true_prediction[0])

np.random.seed(100)
weights_hid = 2 * np.random.random((layer_inp_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1
print(weights_hid)

num_epoch = 150
learning_rate = 0.0001

# prediction_hid = relu(neural_networks(inp[0], weights_hid))
# print(prediction_hid)
# prediction_out = neural_networks(prediction_hid, weights_out)
# print(prediction_out)

for _ in range(num_epoch):
    out_error = 0
    for i in range(len(inp)):
        current_inp = inp[i:i+1]
        prediction_hid = relu(np.dot(current_inp, weights_hid))
        prediction_out = neural_networks(prediction_hid, weights_out)
        out_error += get_error(true_prediction[i:i+1], prediction_out)
        delta_out = prediction_out - true_prediction[i:i+1]
        delta_hid = delta_out.dot(weights_out.T) * reluderiv(prediction_hid)
        print(delta_hid.shape)
        weights_out -= learning_rate * prediction_hid.T.dot(delta_out)
        weights_hid -= learning_rate * current_inp.T.dot(delta_hid)
        print("Prediction_out: %s, True_prediction: %s" % (prediction_out, true_prediction[i:i + 1]))
    print("Errors: %s" %(out_error))
    print("------------------------------------------")
