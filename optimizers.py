import numpy as np


""" Super Class """
class Optimizer(object):
    """
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        pass


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                if n not in self.velocity:
                    self.velocity[n] = -self.lr * dv
                else:
                    self.velocity[n] = self.momentum * self.velocity[n] - self.lr * dv
                layer.params[n] += self.velocity[n]


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                if n not in self.cache:
                    self.cache[n] = (1-self.decay) * np.square(dv)
                else:
                    self.cache[n] = self.decay * self.cache[n] + (1-self.decay) * dv * dv
                layer.params[n] -= self.lr * dv / np.sqrt(self.cache[n]+self.eps) 


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        self.t += 1
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                if n not in self.mt:
                    self.mt[n] = (1-self.beta1) * dv
                    self.vt[n] = (1-self.beta2) * dv * dv
                else:
                    self.mt[n] = self.beta1 * self.mt[n] + (1-self.beta1) * dv
                    self.vt[n] = self.beta2 * self.vt[n] + (1-self.beta2) * dv * dv
                mt_hat = self.mt[n] / (1-self.beta1**self.t)
                vt_hat = self.vt[n] / (1-self.beta2**self.t)
                layer.params[n] -= self.lr * mt_hat / (np.sqrt(vt_hat)+self.eps)
