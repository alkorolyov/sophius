import torch

class LSTMRegressor(torch.nn.Module):
    def __init__(self,
                 input_dim=32,
                 hidden_dim=128,
                 num_layers=1,
                 dropout=0.0,
                 lr=1e-3,
                 gamma=0.9,
                 num_epochs=None,
                 **_):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.lr = lr
        self.gamma = gamma
        self.val_loss = None

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])