import torch
from sophius.encode import Encoder
from sophius.templates import ModelTmpl


class LSTMRegressor(torch.nn.Module):
    def __init__(self,
                 input_dim=32,
                 hidden_dim=128,
                 num_layers=1,
                 dropout=0.0,
                 **_):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

class Estimator:
    def __init__(self, path: str = '../data/models/estimator_v2.pth'):
        self.encoder = Encoder()
        self.estimator = torch.load(path).cpu()
        self.estimator.eval()

    def predict_val_acc(self, model_tmpl: ModelTmpl):
        t = torch.tensor(self.encoder.model2vec(model_tmpl), dtype=torch.float32)
        return self.estimator(t).item()