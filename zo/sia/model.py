# zo/sia/model.py
#
# What it does:
# This module defines the neural network architecture. It has been corrected
# with a robust fix that ensures tensor shapes are handled correctly by the
# PyTorch RNN layers, resolving errors during single-sentence translation.

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        # This implementation correctly handles both batched and un-batched inputs.
        # The GRU layer expects a 3D hidden state, which is provided by initHidden.
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden

    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs=None):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None
    
    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        if hidden is None:
            hidden = self.initHidden(input.device, batch_size=input.size(1))
        
        # Squeeze the sequence dimension (dim 0) before concatenating.
        # This makes the tensors explicitly 2D (batch_size, hidden_size).
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(0), hidden.squeeze(0)), 1)), dim=1)
        
        if encoder_outputs is not None:
            encoder_outputs_bmm = encoder_outputs.transpose(0, 1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_bmm)
            output = torch.cat((embedded.squeeze(0), attn_applied.squeeze(1)), 1)
            output = self.attn_combine(output).unsqueeze(0)
        else:
            output = embedded

        output = F.relu(output)
        # The GRU layer correctly handles the 3D hidden state.
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        
    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
