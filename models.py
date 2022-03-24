"""
This file contains various models used throughout the project
"""
import torch
import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, config, hidden_layers):
        super().__init__()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.loss_fn = nn.MSELoss()

        self.layers = []
        layer_sizes = [config["input_dim"]] + hidden_layers

        for layer_index in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[layer_index - 1],
                                         layer_sizes[layer_index]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_sizes[-1], config["output_dim"]))

        self.layers = nn.Sequential(*self.layers)

        self.config = config
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
        self.log("val_loss", loss, on_epoch=True, on_step=True)


class GRU(pl.LightningModule):
    def __init__(self, config, hidden_dim, layer_dim, dropout_prob):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]

        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim

        # Training parameters
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.loss_fn = nn.MSELoss()

        # GRU layers
        self.gru = nn.GRU(
            self.input_dim, self.hidden_dim, self.layer_dim,
            batch_first=True, dropout=self.dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.config = config
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
        inputs = inputs.view([inputs.shape[0], -1, self.input_dim])

        outputs = self(inputs)

        loss = self.loss_fn(targets, outputs)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        inputs = inputs.view([inputs.shape[0], -1, self.input_dim])

        outputs = self(inputs)
        loss = self.loss_fn(targets, outputs)
        self.log("val_loss", loss, on_epoch=True, on_step=True)
