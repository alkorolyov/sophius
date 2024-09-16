import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class LSTMRegressor(pl.LightningModule):
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
        self.save_hyperparameters()  # Save hyperparameters
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

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        targets = targets.view(-1, 1)
        loss = F.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        targets = targets.view(-1, 1)
        loss = F.mse_loss(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        self.val_loss.append(loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_loss = []

    def on_validation_epoch_end(self):
        avg_val_loss = torch.mean(torch.stack(self.val_loss))
        # print('val', avg_val_loss, len(self.val_loss))
        self.log('hp_metric', avg_val_loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.gamma)
        }
