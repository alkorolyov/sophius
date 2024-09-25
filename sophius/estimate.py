import torch
from sophius.encode import Encoder

encoder = Encoder()
estimator = torch.load('../data/models/estimator_v2.pth').cpu()
estimator.eval()

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

def estimate_val_acc(model_tmpl):
    t = torch.tensor(encoder.model2vec(model_tmpl), dtype=torch.float32)
    return estimator.cpu()(t).item()
