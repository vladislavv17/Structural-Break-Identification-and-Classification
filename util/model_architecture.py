import numpy as np
import torch
from torch import nn


# Вспомогательный класс для получения выхода LSTM слоя
class ExtractLSTMOutput(nn.Module):
    def forward(self, x):
        tensor = x
        return tensor[:, -1, :]


class StackedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StackedLSTM, self).__init__()

        hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5 = [hidden_dim // (2 ** i) for i in
                                                                                range(4, -1, -1)]

        self.lstm_1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim_1, num_layers=10, dropout=0.15)
        self.lstm_2 = nn.LSTM(input_size=input_dim * hidden_dim_1, hidden_size=hidden_dim_2, num_layers=10,
                              dropout=0.15)
        self.lstm_3 = nn.LSTM(input_size=input_dim * hidden_dim_2, hidden_size=hidden_dim_3, num_layers=10,
                              dropout=0.15)
        self.lstm_4 = nn.LSTM(input_size=input_dim * hidden_dim_3, hidden_size=hidden_dim_4, num_layers=10,
                              dropout=0.15)
        self.lstm_5 = nn.LSTM(input_size=input_dim * hidden_dim_4, hidden_size=hidden_dim_5, num_layers=10,
                              dropout=0.15)

    def forward(self, input):
        output, (h_n, c_n) = self.lstm_1(input)
        output, (h_n, c_n) = self.lstm_2(output)
        output, (h_n, c_n) = self.lstm_3(output)
        output, (h_n, c_n) = self.lstm_4(output)
        output, (h_n, c_n) = self.lstm_5(output)

        return output


class SingleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim=1):
        super(SingleLSTM, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, dropout=0.15)

    def forward(self, input):
        output, (h_n, c_n) = self.lstm_1(input)
        return output


class AdvancedLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1, random_seed=42):
        super(AdvancedLSTM, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.net = nn.Sequential(
            # nn.LSTM(input_dim, hidden_dim, layer_dim),
            StackedLSTM(input_dim, hidden_dim),
            ExtractLSTMOutput(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4, hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 16, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Базовая модель с LSTM и линейным слоем
class BaseLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1, random_seed=42):
        super(BaseLSTM, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.net = nn.Sequential(
            SingleLSTM(input_dim, hidden_dim, layer_dim),
            ExtractLSTMOutput(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4, hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 16, output_dim)
        )

    def forward(self, x):
        return self.net(x)
