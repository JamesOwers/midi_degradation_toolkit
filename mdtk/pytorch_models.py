import torch
import torch.autograd as autograd
import torch.nn as nn



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

    def forward(self, batch):
        batch_size = batch.shape[0]
        self.hidden = self.init_hidden(batch_size)
        # Weirdly have to permute batch dimension to second for LSTM...
        embeds = self.embedding(batch).permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
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

    def forward(self, batch):
        batch_size = batch.shape[0]
        self.hidden = self.init_hidden(batch_size)
        # Weirdly have to permute batch dimension to second for LSTM...
        embeds = self.embedding(batch).permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)

        return output



class ErrorIdentificationNet(nn.Module):
    """
    Baseline model for the Error Identification task, in which the label for
    each data point is a binary label for each frame of input, with  0 = not
    degraded and 1 = degraded.
    """
    def __init__(self):
        super().__init__()



class ErrorCorrectionNet(nn.Module):
    """
    Baseline model for the Error Correction task, in which the label for each
    data point is the clean data.
    """
    def __init__(self):
        super().__init__()
