"""
This file contains various models used throughout the project
"""
import torch
import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, inputs=12, outputs=1, lr=1e-3, loss_fn=nn.MSELoss()):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Linear(inputs, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, outputs)
            )

        self.lr = lr
        self.loss_fn = loss_fn

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = self.loss_fn(targets, outputs)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self(inputs)
        loss = self.loss_fn(targets, outputs)
        self.log("val_loss", loss)


class GRU(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,
                 dropout_prob, loss_fn=nn.MSELoss(), batch_size=1, lr=1e-3):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        self.input_dim = input_dim
        self.batch_size = batch_size

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.lr = lr
        self.loss_fn = loss_fn

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,
                         device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of
        # (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape
        # (batch_size, output_dim)
        out = self.fc(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        inputs = inputs.view([self.batch_size, -1, self.input_dim])

        outputs = self(inputs)

        loss = self.loss_fn(targets, outputs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        inputs = inputs.view([self.batch_size, -1, self.input_dim])

        outputs = self(inputs)
        loss = self.loss_fn(targets, outputs)
        self.log("val_loss", loss)
