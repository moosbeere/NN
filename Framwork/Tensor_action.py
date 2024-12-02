import numpy as np

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
                if (self.operation_on_creation == "-1"):
                    self.creators[0].backward(self.grad.__neg__(), self)
                if (self.operation_on_creation == "-"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(),self)
                if (self.operation_on_creation == "*"):
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0],self)

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

    def check_grads_from_child(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

a_1 = Tensor([1,2,3], autograd=True)
a_2 = Tensor([4,5,6],autograd=True)
a_4 = Tensor([9,8,7],autograd=True)

a_mult = a_1 * a_2
a_add_1 = a_1 + a_2
a_add_2 = a_2 - a_4
# print(a_add_1)
# print(a_add_2)
a_add_3 = a_add_2 + a_add_1
# a_add_3.backward(Tensor([4,5,6]))
a_mult.backward(Tensor([4,5,6]))

print(f"a_1: {a_1.grad}")
print(f"a_2: {a_2.grad}")
# print(f"a_4: {a_4.grad}")


