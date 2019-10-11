import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np



class Command_ErrorDetectionNet(nn.Module):
    """
    Baseline model for the Error Detection task, in which the label for each
    data point is either 1 (degraded) or 0 (not degraded).
    Adapted from: https://github.com/claravania/lstm-pytorch/blob/master/model.py
    It:
    1)    embeds the integer batch input into a learned embedding space
    2)    passes this through a standard LSTM with one hidden layer
    3)    passes the final hidden state from the lstm through a dropout layer
    4)    then puts this through a linear layer and returns the output
    
    You should use nn.CrossEntropyLoss which will perform both a softmax on
    the output, then Negative log likelihood calculation (this is more
    efficient and therefore I exclude a softmax layer from the model)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size,
                 dropout_prob=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=dropout_prob)


    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_dim),
                torch.randn(1, batch_size, self.hidden_dim))

    def forward(self, batch, input_lengths=None):
        if input_lengths is not None:
            batch_length = np.max(input_lengths)
            batch = batch[:, :batch_length]
        batch_size = batch.shape[0]
        self.hidden = self.init_hidden(batch_size)
        # Weirdly have to permute batch dimension to second for LSTM...
        embeds = self.embedding(batch).permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        if input_lengths is None:
            out = ht[-1]
        else:
            out = outputs[input_lengths - 1, np.arange(batch_size)]
        output = self.dropout_layer(out)
        output = self.hidden2out(output)

        return output
        


class Command_ErrorClassificationNet(nn.Module):
    """
    Baseline model for the Error Classification task, in which the label for
    each data point is a degradation_id (with 0 = not degraded).
    Adapted from: https://github.com/claravania/lstm-pytorch/blob/master/model.py
    It:
    1)    embeds the integer batch input into a learned embedding space
    2)    passes this through a standard LSTM with one hidden layer
    3)    passes the final hidden state from the lstm through a dropout layer
    4)    then puts this through a linear layer and returns the output
    
    You should use nn.CrossEntropyLoss which will perform both a softmax on
    the output, then Negative log likelihood calculation (this is more
    efficient and therefore I exclude a softmax layer from the model)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size,
                 dropout_prob=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=dropout_prob)


    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_dim),
                torch.randn(1, batch_size, self.hidden_dim))

    def forward(self, batch, input_lengths=None):
        if input_lengths is not None:
            batch_length = np.max(input_lengths)
            batch = batch[:, :batch_length]
        batch_size = batch.shape[0]
        self.hidden = self.init_hidden(batch_size)
        # Weirdly have to permute batch dimension to second for LSTM...
        embeds = self.embedding(batch).permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        if input_lengths is None:
            out = ht[-1]
        else:
            out = outputs[input_lengths - 1, np.arange(batch_size)]
        output = self.dropout_layer(out)
        output = self.hidden2out(output)

        return output



class Pianoroll_ErrorIdentificationNet(nn.Module):
    """
    Baseline model for the Error Identification task, in which the label for
    each data point is a binary label for each frame of input, with  0 = not
    degraded and 1 = degraded.
    
    The model consists of:
    1) A bidirectional LSTM.
    2) A sequence of dropout layers followed by linear layers.
    3) A final dropout layer.
    4) A final output layer of dim 2.
    
    The outputs and labels should be flattened when computing the CE Loss.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers=[], dropout_prob=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        
        current_dim = 2 * hidden_dim
        linear_list = []
        for dim in layers:
            linear_list.append(nn.Dropout(p=dropout_prob))
            linear_list.append(nn.Linear(current_dim, dim))
            current_dim = dim
        
        self.linears = nn.ModuleList(linear_list)
        
        self.hidden2out = nn.Linear(current_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim),
                torch.randn(2, batch_size, self.hidden_dim))
        
    def forward(self, batch):
        batch_size = batch.shape[0]
        self.hidden = self.init_hidden(batch_size)
        # Weirdly have to permute batch dimension to second for LSTM...
        batch = batch.permute(1, 0, 2)
        output, _ = self.lstm(batch.float(), self.hidden)
        output = output.permute(1, 0, 2)
        
        for module in self.linears:
            output = module(output)
        
        output = self.dropout_layer(output)
        output = self.hidden2out(output)

        return output



class ErrorCorrectionNet(nn.Module):
    """
    Baseline model for the Error Correction task, in which the label for each
    data point is the clean data.
    """
    def __init__(self):
        super().__init__()
