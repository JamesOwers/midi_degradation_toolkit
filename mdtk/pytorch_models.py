import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_size=2,
        dropout_prob=0.1,
        num_lstm_layers=1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_lstm_layers)

        self.hidden2out = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=dropout_prob)

    def init_hidden(self, batch_size, device):
        return (
            torch.randn(1, batch_size, self.hidden_dim, device=device),
            torch.randn(1, batch_size, self.hidden_dim, device=device),
        )

    def forward(self, batch, input_lengths=None):
        if input_lengths is not None:
            batch_length = np.max(input_lengths)
            batch = batch[:, :batch_length]
        batch_size = batch.shape[0]
        device = batch.device
        self.hidden = self.init_hidden(batch_size, device=device)
        embeds = self.embedding(batch).permute(1, 0, 2)
        #        embeds = self.embedding(batch)
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


class Command_ErrorClassificationNet(Command_ErrorDetectionNet):
    """
    Baseline model for the Error Classification task, in which the label for
    each data point is a degradation_id (with 0 = not degraded).

    It's precisely the same network design as for task 1 - error detection,
    except this has a number of output classes (9 for ACME1.0).
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_size=9,
        dropout_prob=0.1,
        num_lstm_layers=1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_size=output_size,
            dropout_prob=dropout_prob,
            num_lstm_layers=num_lstm_layers,
        )


class Pianoroll_ErrorLocationNet(nn.Module):
    """
    Baseline model for the Error Location task, in which the label for
    each data point is a binary label for each frame of input, with  0 = not
    degraded and 1 = degraded.

    The model consists of:
    1) A bidirectional LSTM.
    2) A sequence of dropout layers followed by linear layers.
    3) A final dropout layer.
    4) A final output layer of dim 2.

    The outputs and labels should be flattened when computing the CE Loss.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        layers=[],
        dropout_prob=0.1,
        num_lstm_layers=1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        current_dim = 2 * hidden_dim
        linear_list = []
        for dim in layers:
            linear_list.append(nn.Dropout(p=dropout_prob))
            linear_list.append(nn.Linear(current_dim, dim))
            linear_list.append(nn.ELU())
            current_dim = dim

        self.linears = nn.ModuleList(linear_list)

        self.hidden2out = nn.Linear(current_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout_prob)

    def init_hidden(self, batch_size, device):
        return (
            torch.randn(2, batch_size, self.hidden_dim, device=device),
            torch.randn(2, batch_size, self.hidden_dim, device=device),
        )

    def forward(self, batch):
        batch_size = batch.shape[0]
        device = batch.device
        output, _ = self.lstm(batch.float(), self.init_hidden(batch_size, device))

        for module in self.linears:
            output = module(output)

        output = self.dropout_layer(output)
        output = self.hidden2out(output)

        return output


class Pianoroll_ErrorCorrectionNet(nn.Module):
    """
    Baseline model for the Error Correction task, in which the label for each
    data point is the clean data.

    The model consists of:
    1) Bi-LSTM to embed the input.
    2) A linear connection layer.
    3) A 2nd Bi-LSTM to decode.
    4) Final output layers.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        layers=[],
        dropout_prob=0.1,
        num_lstm_layers=1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.connector = nn.Linear(hidden_dim * 2, hidden_dim)
        self.connector_do = nn.Dropout(p=dropout_prob)

        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        current_dim = 2 * hidden_dim
        linear_list = []
        for dim in layers:
            linear_list.append(nn.Dropout(p=dropout_prob))
            linear_list.append(nn.Linear(current_dim, dim))
            linear_list.append(nn.ELU())
            current_dim = dim

        self.linears = nn.ModuleList(linear_list)

        self.hidden2out = nn.Linear(current_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout_prob)

    def init_hidden(self, batch_size, device):
        return (
            torch.randn(2, batch_size, self.hidden_dim, device=device),
            torch.randn(2, batch_size, self.hidden_dim, device=device),
        )

    def forward(self, batch, input_lengths):
        batch_size = batch.shape[0]
        device = batch.device
        output, _ = self.encoder(batch.float(), self.init_hidden(batch_size, device))

        output = self.connector_do(F.elu(self.connector(output)))

        output, _ = self.decoder(output, self.init_hidden(batch_size, device))

        for module in self.linears:
            output = module(output)

        output = self.dropout_layer(output)
        output = self.hidden2out(output)

        return torch.sigmoid(output)
