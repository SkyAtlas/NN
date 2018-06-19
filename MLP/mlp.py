import numpy as np
from common.activate_function import sigmoid 

class MLP:

    def __init__(self, i_nodes, h_nodes, o_nodes, eta):
    # TODO:: describe
        self.input_nodes = i_nodes
        self.hidden_nodes = h_nodes
        self.output_nodes = o_nodes
        self.eta = eta
        self.W_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5))
        self.W_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5))
        self.activate_function = lambda X: sigmoid(X)
        pass

    def fit(self):
        pass
    
    def predict(self, X):
        hidden_input = np.dot(self.W_input_hidden, X)
        hidden_output = self.activate_function(hidden_input)

        out_input = np.dot(self.W_hidden_output, hidden_output)
        out_output = self.activate_function(out_input)
        pass
    
