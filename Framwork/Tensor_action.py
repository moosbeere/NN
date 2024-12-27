import numpy as np
from keras.src.models.functional import operation_fn
from tensorflow.compiler.tf2xla.python.xla import transpose


class Tensor(object):
    grad = None
    # children = {}
    def __init__(self, data, creators = None, operation_on_creation = None,  autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.autograd = autograd
        # self.grad = None
        self.children = {}

        if id is None:
            self.id = np.random.randint(0, 100000000)

        if (self.creators is not None):
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] +=1


    def __add__(self, other):
        if self.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        else:
            return Tensor(self.data + other.data)

    def __str__(self):
        # return str(self.data.__str__())
        return str(self.data)

    def backward(self, grad=None, grad_child=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_child is not None:
                if (self.children[grad_child.id]) > 0:
                    self.children[grad_child.id] -=1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if (self.creators is not None and (self.check_grads_from_child() or grad_child is None)):
                if (self.operation_on_creation == "+"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad,self)
                elif (self.operation_on_creation == "-1"):
                    self.creators[0].backward(self.grad.__neg__(), self)
                elif (self.operation_on_creation == "-"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(),self)
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                elif (self.operation_on_creation == "*"):
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0], self)
                elif "dot" in self.operation_on_creation:
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.grad.transpose().dot(self.creators[0]).transpose()
                    self.creators[1].backward(temp,self)
                elif "transpose" in self.operation_on_creation:
                    self.creators[0].backward(self.grad.transpose(), self)
                elif "sigmoid" in self.operation_on_creation:
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(self * (temp - self)), self)
                elif "tanh" in self.operation_on_creation:
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(temp - (self * self)), self)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        else:
            return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        else:
            return Tensor(self.data * other.data)

    def sum(self, axis):
        if(self.autograd):
            return Tensor(self.data.sum(axis), [self], "sum_"+str(axis),True)
        return(self.data.sum())

    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        # print(transpose)
        expand_shape= list(self.data.shape) + [count_copies]
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        expand_data = expand_data.transpose(transpose)
        if (self.autograd):
            return Tensor(expand_data, [self], "expand_"+str(axis), autograd=True)
        return Tensor(expand_data)

    def dot(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        else:
            return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        else:
            return Tensor(self.data.transpose())

    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid", True)
        else:
            return Tensor(1/(1+np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh", True)
        else:
            return Tensor(np.tanh(self.data))

    def check_grads_from_child(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

a_1 = Tensor([[1,2,3],[4,5,6]], autograd=True)
a_2 = a_1.sigmoid()
a_2.backward(Tensor([4,5,10]))
print(a_2)
print(a_1.grad)

a_3 = Tensor([[2,3,4],[2,3,5]], autograd=True)
a_4 = a_3.tanh()
a_4.backward(Tensor([4,5,10]))
print(a_3.grad)







