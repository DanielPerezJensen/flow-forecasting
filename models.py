"""
This file contains various models used throughout the project
"""
import torch
import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, hidden_layers, name="MLP", inputs=12, outputs=1,
                 lr=1e-3, weight_decay=1e-6, loss_fn=nn.MSELoss(), lag=6,
                 scaler=None, time_features=False, index_features=False,
                 index_area_features=False, index_cloud_features=False):
        super().__init__()

        self.name = name

        # Defining some parameters about this model
        self.time_features = time_features
        self.index_features = index_features

        self.layers = []
        layer_sizes = [inputs] + hidden_layers

        for layer_index in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[layer_index - 1],
                                         layer_sizes[layer_index]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_sizes[-1], outputs))

        self.layers = nn.Sequential(*self.layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
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
                 dropout_prob, name="GRU", loss_fn=nn.MSELoss(), batch_size=1,
                 weight_decay=1e-6, lr=1e-3, lag=6, scaler=None,
                 time_features=False, index_features=False,
                 index_area_features=False, index_cloud_features=False):
        super().__init__()

        self.name = name

        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim

        # Defining some parameters about this model
        self.time_features = time_features
        self.index_features = index_features

        self.batch_size = batch_size

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

        self.save_hyperparameters()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
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
