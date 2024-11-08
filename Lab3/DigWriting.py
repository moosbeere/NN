import numpy as np
from keras.src.datasets import mnist
from mnist import train_images

from pythonProject.Lab3.ReLU import weights_out, weights_hid


def relu(x):
    return (x > 0) * x

def reluderiv(x):
    return (x > 0)

train_images_count = 1000
test_images_count = 10000
pixels_per_img = 28 * 28
digits_num = 10

(x_train, y_train),(x_test, y_test) = mnist.load_data()
train_images = x_train[0:train_images_count].reshape(train_images_count, pixels_per_img)/255
train_labels = y_train[0:train_images_count]
test_images = x_test[0:test_images_count].reshape(test_images_count, pixels_per_img)/255
test_labels = y_test[0:test_images_count]

one_hot_labels = np.zeros((len(train_labels), digits_num))
for i in range(len(one_hot_labels)):
    one_hot_labels[i][train_labels[i]] = 1
train_labels = one_hot_labels
# print(train_labels)

one_hot_labels = np.zeros((len(test_labels), digits_num))
for i in range(len(one_hot_labels)):
    one_hot_labels[i][test_labels[i]] = 1
test_labels = one_hot_labels

np.random.seed(2)

hidden_size = 150
weight_hid = 0.2 * np.random.random((pixels_per_img, hidden_size)) - 0.1
weight_out = 0.2 * np.random.random((hidden_size, digits_num)) - 0.1

learning_rate = 0.01
num_epoch = 500
batch_size = 50

for i in range(num_epoch):
    correct_answer = 0
    for j in range(int(len(train_images)/batch_size)):
        batch_start = batch_size * j
        batch_end = batch_size * (j + 1)
        layer_in = train_images[batch_start:batch_end]
        layer_hid = relu(layer_in.dot(weight_hid))
        dropout_mask = np.random.randint(3,size = layer_hid.shape)# случайное число от 0 до 1 (максимальное число 2) с размером равному скрытому слою
        layer_hid *= dropout_mask * 2
        layer_out = layer_hid.dot(weight_out)
        for k in range(batch_size):
            correct_answer += int(np.argmax(layer_out[k:k+1]) == np.argmax(train_labels[batch_start + k:batch_start  + k +1]))
        layer_out_delta = (layer_out - train_labels[batch_start:batch_end])/batch_size
        layer_hid_delta = layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid) * dropout_mask
        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)
        print("Epoch: ", i+1)
        print("Accuracy: %.2f" %(correct_answer * 100/len(train_images)))

correct_answer = 0
for j in range(len(test_images)):
        layer_in = test_images[j:j+1]
        layer_hid = relu(layer_in.dot(weight_hid))
        layer_out = layer_hid.dot(weight_out)
        correct_answer += int(np.argmax(layer_out) == np.argmax(test_labels[j:j+1]))
print("Accuracy: %.2f" % (correct_answer * 100 / len(test_images)))
