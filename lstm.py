import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_cols, features=115, filters=128, dropout=0.5, maxpool=2, kernel_size=6):
        super(CNN, self).__init__()
        lin_features = int((num_cols - (2*(kernel_size - 1))) / maxpool)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(maxpool),
            nn.Linear(in_features=lin_features, out_features=features),
        )

    def forward(self, x):
        return self.cnn(x)

    def freeze(self, warmup=False):
        self.cnn.requires_grad_(False)

class LSTM(nn.Module):
    def __init__(self, num_classes, features=115, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=features, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=features, out_features=num_classes)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        results = self.dense(hn[-1])
        return results

    def freeze(self, warmup=False):
        self.lstm.requires_grad_(False)

class CNNLSTM(nn.Module):
    def __init__(self, num_cols, num_classes, features=115, filters=128, dropout=0.5, maxpool=2, kernel_size=6, bidirectional=False):
        super(CNNLSTM, self).__init__()
        self.cnn = CNN(num_cols, features=features, filters=filters, dropout=dropout, maxpool=maxpool, kernel_size=kernel_size)
        self.lstm = LSTM(num_classes, features, bidirectional=bidirectional)

    def forward(self, x):
        x = self.cnn(x)
        return self.lstm(x)

    def freeze(self, warmup=False):
        self.cnn.freeze(warmup=warmup)
        if warmpup:
            self.lstm.freeze(warmup=warmup)
