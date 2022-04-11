import logging
import numpy as np
import utils
import sys
import itertools

logging.basicConfig(
    level=logging.INFO, 
    format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("{0}.log".format("nn")),
        logging.StreamHandler(sys.stdout)
    ]
)

# Activation functions
def softmax(X):
    # Normalize the softmax 
    # https://deepnotes.io/softmax-crossentropy
    e = np.exp(X - X.max(1))
    r = e / e.sum(1)
    return r
def delta_cross_entropy(Y, G):
    return (Y - G) / Y.shape[0]
def relu(X):
    return np.maximum(X, 0)
def relu_deriv(Y, G):
    return np.multiply((1. * (Y > 0)), G)

class NNBuilder(object):
    def __init__(self):
        self.layers = []
    def load(self, Sigmoid, Sigmoid_deriv, nn_path, CLS, input_dim):
        B,W = utils.read_nn(nn_path, input_dim)
        if not(len(B) == len(W) and len(B) == len(CLS) and len(B) == len(Sigmoid) and len(B) == len(Sigmoid_deriv)):
            raise RuntimeError("Must be describing the same network.")
        for b,w,cls,sigmoid,sigmoid_deriv in zip(B,W,CLS,Sigmoid,Sigmoid_deriv):
            self.layers.append(cls(sigmoid, sigmoid_deriv, (b,w)))
        return self.build()
    def build_all(self, Sigmoid, Sigmoid_deriv, CLS, Dim):
        for cls,sigmoid,sigmoid_deriv,dim in zip(CLS,Sigmoid,Sigmoid_deriv,Dim):
            self.layers.append(cls(sigmoid, sigmoid_deriv, cls.rand_init_layer(*dim)))
        return self.build()
    def build(self):
        before_layer = None
        after_layer = None
        len_layers = len(self.layers)
        logging.info("Building Neural Network with %d layers.", len_layers)
        for i in range(len_layers):
            # The case for which the current layer is the last layer
            if (i+1) < len_layers:
                after_layer = self.layers[i+1]
            else:
                after_layer = None
            self.layers[i].bind(after_layer)
            logging.info("Binding %s with before(%s) and after(%s)", self.layers[i], before_layer, after_layer)
            before_layer = self.layers[i]
        return self.layers[0]
    def add(self, layer):
        self.layers.append(layer)
        return self

# Parent Class for all layers
class Layer(object):
    # n is the size of the input, m is the size of the output
    def __init__(self, sigmoid, sigmoid_deriv, init_params, name=None):
        self.n,self.m = self.init_layer(init_params)
        if name == None:
            self.name = "{}-{}".format(self.n, self.m)
        else:
            self.name = name
        self.sigmoid = sigmoid
        self.sigmoid_deriv = sigmoid_deriv
    def bind(self, after_layer):
        self.after_layer = after_layer
    def predict(self, X):
        # Peform sanity check
        self.check_type(X)
        l = self.forward(X)
        return l
    def train(self, X, G, lr=0.01):
        self.check_type(X)
        self.check_type(G)
        self.backward(X, G)
        self.update(lr)
    # Computes and returns the final output
    def forward(self, input_data):
        activation = self.compute_forward(input_data)
        if self.after_layer != None:
            return self.after_layer.forward(activation)
        else:
            return activation
    # Computes and returns the list of gradients
    def backward(self, input_activation, ground_truth):
        output_activation = self.compute_forward(input_activation)
        # In the case of the last layer, the GT is the output gradient
        output_gradient = ground_truth
        # If not we need to get the output gradient from the next layer
        if self.after_layer != None:
            output_gradient = self.after_layer.backward(output_activation, ground_truth)
        # apply the sigmoid deriv before anything else
        output_gradient = self.sigmoid_deriv(output_activation, output_gradient)
        # Compute, but not update the parameter gradients
        self.compute_param_gradient(input_activation, output_activation, output_gradient)
        # Return the gradient to the previous layer
        return self.compute_input_gradient(input_activation, output_activation, output_gradient)
    def update(self, lr):
        self.compute_update(lr)
        if self.after_layer != None:
            return self.after_layer.update(lr)
    def get_cost(self, input_data, ground_truth):
        activation = self.compute_forward(input_data)
        if self.after_layer != None:
            return self.after_layer.get_cost(activation, ground_truth)
        else:
            return self.compute_cross_entropy_cost(activation, ground_truth)
    def get_nn_deriv(self):
        if self.after_layer == None:
            return [ self.deriv() ]
        else:
            return [ self.deriv() ] + self.after_layer.get_nn_deriv()
    def compute_forward(self, input_data):
        raise NotImplementedError
    def init_layer(self):
        raise NotImplementedError
    def check_type(self, D):
        if not (isinstance(D, np.matrix) and D.dtype == np.float128):
            raise RuntimeError("Must be Matrix and must be of float 128bits")
    def __str__(self):
        return self.name
    def __str__(self):
        return self.name

class FullyConnectedLayer(Layer):
    # Start NN from scratch
    def init_layer(self, init_params):
        self.b,self.W = init_params
        self.check_type(self.b)
        self.check_type(self.W)
        return self.W.shape
    @staticmethod
    def rand_init_layer(n, m):
        # Follow https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
        b = np.asmatrix(np.zeros(m).astype(np.float128))
        W = np.asmatrix(np.random.randn(n, m).astype(np.float128) * np.sqrt(2.0/m))
        return b,W
    def compute_forward(self, X):
        return self.sigmoid(X * self.W + self.b)
    def compute_input_gradient(self, X, Y, G):
        return G * self.W.T
    def compute_param_gradient(self, X, Y, G):
        # We apply the derivative for the sigmoid first
        self.db = np.sum(G, axis=0)
        self.dW = X.T.dot(G)
    # Not supposed to be here, but lol.
    def compute_cross_entropy_cost(self, Y, G):
        return - np.multiply(G, np.log(Y)).sum() / Y.shape[0]
        #return - np.multiply(G,np.log(Y)) + np.multiply((1-G), np.log(1-Y)).sum() / Y.shape[0]
    def compute_update(self, lr):
        # We attempt to update the b then the W in place, in memory
        # We group them so we can use one loop to update all of them
        params_iter = itertools.chain(np.nditer(self.b, op_flags=['readwrite']), 
                                      np.nditer(self.W, op_flags=['readwrite']))
        params_delta_iter = itertools.chain(np.nditer(self.db), np.nditer(self.dW))
        for param, params_delta in zip(params_iter, params_delta_iter):
            param -= lr * params_delta
    def deriv(self):
        return (self.db, self.dW)
