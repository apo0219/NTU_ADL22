from typing import Dict

import torch
from torch.nn import RNN
from torch.nn import LSTM
from torch.nn import GRU
from torch.nn import Linear
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Softmax
RNN_ID = 0
LSTM_ID = 1
GRU_ID = 2
class Slot_tag(torch.nn.Module):
    def __init__(self, input_size, hidden_size, droput, output_size, batch_size, num_layers, model_type):
        super(Slot_tag, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.droput = droput
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.model_type = model_type
        if (self.model_type == RNN_ID):
            self.rnn = RNN(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True, bidirectional = True)
        elif (self.model_type == LSTM_ID):
            self.lstm = LSTM(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True, bidirectional = True)
        else:
            self.gru = GRU(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True, bidirectional = True) 
        
        self.hidden2out = Linear(self.hidden_size * 2, output_size)
        self.sm = Softmax(dim = 2)

    def forward(self, data_in):
        if (self.model_type == RNN_ID):
            packed_nn_out, _ = self.rnn(data_in)
        elif (self.model_type == LSTM_ID):
            packed_nn_out, _ = self.lstm(data_in)
        else:
            packed_nn_out, _ = self.gru(data_in)
        
        unpacked_rnn_out = pad_packed_sequence(packed_nn_out, batch_first = True)
        linear_output = self.hidden2out(unpacked_rnn_out[0])
        rtn = self.sm(linear_output)
        return rtn

class Intent_cls(torch.nn.Module):
    def __init__(self, input_size, hidden_size, droput, output_size, batch_size, num_layers, model_type):
        super(Intent_cls, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.droput = droput
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.model_type = model_type
        if (self.model_type == RNN_ID):
            self.rnn = RNN(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True)
        elif (self.model_type == LSTM_ID):
            self.lstm = LSTM(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True)
        else:
            self.gru = GRU(input_size = self.input_size, hidden_size = hidden_size, dropout = self.droput, num_layers = self.num_layers, batch_first = True) 
        
        self.hidden2out = Linear(self.hidden_size, output_size)
        self.sm = Softmax(dim = 1)

    def forward(self, data_in):
        if (self.model_type == RNN_ID):
            packed_nn_out, _ = self.rnn(data_in)
        elif (self.model_type == LSTM_ID):
            packed_nn_out, _ = self.lstm(data_in)
        else:
            packed_nn_out, _ = self.gru(data_in)

        unpacked_rnn_out = pad_packed_sequence(packed_nn_out, batch_first = True)
        linear_input = torch.stack([unpacked_rnn_out[0][i][unpacked_rnn_out[1][i] - 1] for i in range(self.batch_size)])
        linear_output = self.hidden2out(linear_input)
        rtn = self.sm(linear_output)
        return rtn

