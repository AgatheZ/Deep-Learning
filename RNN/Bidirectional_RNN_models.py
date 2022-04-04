import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        #for input
        self.x2h = nn.Linear(input_size, 4*hidden_size, bias=bias)

        #for cell state/hidden state 
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
            
        # We used hx to pack both the hidden and cell states
        hx, cx = hx
        x = self.x2h(input)
        ht = self.h2h(hx)
        
        concatenate =  x + ht
        ft, it, ct, ot = torch.tensor_split(concatenate, 4, dim=1)
        #print(torch.tensor_split(concatenate, 4, dim=1).shape)

        #update of the forget gate ft
        sig = nn.Sigmoid()
        th = nn.Tanh()

        ft = sig(ft)

        #update of the input gate
        it = sig(it)

        #update of the output gate 
        ot = sig(ot)

        #candidate cell state update
        ct = th(ct)

        #actual cell state update
        cy = ft * cx + it * ct

        #update of the hidden state 
        hy = ot * th(cy)

        return (hy, cy)

#Returns the updated hidden state given the previous hidden state 
class BasicRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(BasicRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.reset_parameters()
        

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

            
    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        activation = getattr(nn.functional, self.nonlinearity)
        hy = activation(self.x2h(input) + self.h2h(hx))

        return hy

    
    
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2r = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2n = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2n = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()
        

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        sig = nn.Sigmoid()
        th = nn.Tanh()
        zt = sig(self.x2h(input) + self.h2h(hx))
        rt = th(self.x2r(input) + self.h2r(hx))

        inpt_ht = th(self.h2n(rt*hx) + self.x2n(input))
        hy = (1 - zt) * hx + zt * inpt_ht
        
        return hy

class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.fc = nn.Linear(2*self.hidden_size, self.output_size)
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list_rev = nn.ModuleList()

        if self.mode == 'LSTM': 
            self.rnn_cell_list.append(LSTMCell(self.input_size, self.hidden_size, self.bias))
            for i in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size, self.bias))

        if self.mode == 'GRU':
            self.rnn_cell_list.append(GRUCell(self.input_size, self.hidden_size, self.bias))
            for i in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size, self.hidden_size, self.bias))
        
        if self.mode == 'RNN_TANH':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size, self.hidden_size, self.bias))
            for i in range(1,self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size, self.hidden_size, self.bias)) 

        if self.mode == 'RNN_RELU':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size, self.hidden_size, self.bias, "relu"))
            for i in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size, self.hidden_size, self.bias, "relu"))    
        
    def forward(self, input, hx=None):

        if hx == None:
            h0 = input.new_zeros((self.num_layers, input.size(0), self.hidden_size), requires_grad=False)
        else: 
            h0 = list(hx)
        
        outs = []
        outs_rev = []
        
        if self.mode == 'LSTM':
            forward_layers = [[h0[i], h0[i]] for i in range(self.num_layers)]
            backward_layers = [[h0[i], h0[i]] for i in range(self.num_layers)]

            for t in range(input.shape[1]):
                forward_layers[0] = self.rnn_cell_list[0](input[:, t, :], forward_layers[0])
                #the input sequence is fed in the opposite order
                backward_layers[0] = self.rnn_cell_list[0](input[:, input.shape[1] - t - 1, :], backward_layers[0])
                
                for hidden_layer in range(1, self.num_layers):
                    forward_layers[hidden_layer] = self.rnn_cell_list[hidden_layer](forward_layers[hidden_layer - 1][0], forward_layers[hidden_layer])
                    backward_layers[hidden_layer] = self.rnn_cell_list[hidden_layer](backward_layers[hidden_layer - 1][0], backward_layers[hidden_layer])
                
                outs.append(forward_layers[-1][0])
                outs_rev.append(backward_layers[-1][0])

        else:
            forward_layers = [h0[i] for i in range(self.num_layers)]
            backward_layers = [h0[i] for i in range(self.num_layers)]

            for t in range(input.shape[1]):
                forward_layers[0] = self.rnn_cell_list[0](input[:, t, :], forward_layers[0])
                
                #the input sequence is fed in the opposite order 
                backward_layers[0] = self.rnn_cell_list[0](input[:, input.shape[1] - t - 1, :], backward_layers[0])

                for hidden_layer in range(1, self.num_layers):
                    forward_layers[hidden_layer] = self.rnn_cell_list[hidden_layer](forward_layers[hidden_layer - 1], forward_layers[hidden_layer])
                    backward_layers[hidden_layer] = self.rnn_cell_list[hidden_layer](backward_layers[hidden_layer - 1], backward_layers[hidden_layer])
            
                outs.append(forward_layers[-1])
                outs_rev.append(backward_layers[-1])    
                
        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)
        out = self.fc(out)
        return out