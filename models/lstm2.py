import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=10, time_lag=14, hidden_num=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.time_lag = time_lag

        self.hidden = self.init_hidden()

        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Conv1d(self.time_lag, 1, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self):
        return (torch.zeros(1, self.time_lag, self.hidden_size),
                torch.zeros(1, self.time_lag, self.hidden_size))

    def forward(self, x):
        x, self.hidden = self.lstm1(x, self.hidden)
        for i in range(self.hidden_num):
            x = self.dropout(x)
            x, self.hidden = self.lstm2(x, self.hidden)

        pred = self.linear(x)

        return pred


class LSTM2(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=10, time_lag=14, hidden_num=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.time_lag = time_lag

        self.hidden = self.init_hidden()

        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self):
        return (torch.zeros(1, self.time_lag, self.hidden_size),
                torch.zeros(1, self.time_lag, self.hidden_size))

    def forward(self, x):
        x, self.hidden = self.lstm1(x, self.hidden)
        for i in range(self.hidden_num):
            x = self.dropout(x)
            x, self.hidden = self.lstm2(x, self.hidden)

        x = x.view(-1, self.hidden_size)
        pred = self.linear(x)

        return pred