"""
This file contains various models used throughout the project
"""
import torch
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score
from omegaconf import DictConfig, OmegaConf

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from typing import Any, Optional, Tuple, List, Dict, Callable, Type, Union

ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]


def shared_eval_step(
    model: pl.LightningModule,
    batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    inputs, targets = batch

    outputs = model(inputs)
    loss = model.loss_fn(targets, outputs)

    return loss, outputs, targets


def shared_eval_epoch_end(
    model: pl.LightningModule, step_outputs: List[Dict[str, torch.Tensor]]
) -> float:
    outputs, targets = [], []

    for out in step_outputs:
        outputs.append(out["outputs"])
        targets.append(out["targets"])

    np_outputs = torch.cat(outputs).cpu().detach().numpy()
    np_targets = torch.cat(targets).cpu().detach().numpy()

    descaled_outputs = model.scaler.inverse_transform(np_outputs)
    descaled_targets = model.scaler.inverse_transform(np_targets)

    scaled_loss = mean_squared_error(descaled_targets, descaled_outputs,
                                     squared=False)

    return scaled_loss


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scaler: ScalerType,
        cfg: DictConfig
    ) -> None:

        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()
        self.scaler = scaler

        self.layers = []  # type: List[nn.Module]
        layer_sizes = [input_dim] + list(self.cfg.model.layers)

        for layer_index in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[layer_index - 1],
                                         layer_sizes[layer_index]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_sizes[-1], output_dim))

        self.model = nn.Sequential(*self.layers)  # type: Callable[[torch.Tensor], torch.Tensor]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, train_batch, batch_idx)
        self.log("train_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, val_batch, batch_idx)
        self.log("val_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, training_step_outputs)

        self.log("train_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, validation_step_outputs)

        self.log("val_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)


class GRU(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scaler: ScalerType,
        cfg: DictConfig
    ) -> None:
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_dim = cfg.model.layer_dim
        self.dropout_prob = cfg.model.dropout_prob
        self.hidden_dim = cfg.model.hidden_dim

        self.loss_fn = nn.MSELoss()

        # GRU/LSTM layers
        self.model = nn.GRU(
            self.input_dim, self.hidden_dim, self.layer_dim,
            batch_first=True, dropout=self.dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)  # type: Callable[[torch.Tensor], torch.Tensor]

        self.scaler = scaler
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,
                         device=x.device).requires_grad_().to(x.device)

        x = x.view([x.shape[0], -1, self.input_dim])

        # Forward propagation by passing in the input and hidden state
        gru_out, _ = self.model(x, h0.detach())

        # Reshaping the outputs in the shape of
        # (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        gru_out = gru_out[:, -1, :]

        # Convert the final state to our desired output shape
        # (batch_size, output_dim)
        return self.fc(gru_out)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, train_batch, batch_idx)
        self.log("train_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, val_batch, batch_idx)
        self.log("val_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, training_step_outputs)

        self.log("train_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, validation_step_outputs)

        self.log("val_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)


class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scaler: ScalerType,
        cfg: DictConfig
    ) -> None:
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_dim = cfg.model.layer_dim
        self.dropout_prob = cfg.model.dropout_prob
        self.hidden_dim = cfg.model.hidden_dim

        self.loss_fn = nn.MSELoss()

        # GRU/LSTM layers
        self.model = nn.LSTM(
            input_dim, cfg.model.hidden_dim, cfg.model.layer_dim,
            batch_first=True, dropout=cfg.model.dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(cfg.model.hidden_dim, output_dim)  # type: Callable[[torch.Tensor], torch.Tensor]

        self.scaler = scaler
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim).requires_grad_().to(x.device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim).requires_grad_().to(x.device)

        x = x.view([x.shape[0], -1, self.input_dim])

        lstm_out, (hn, cn) = self.model(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of
        # so that it can fit into the fully connected layer
        lstm_out = lstm_out[:, -1, :]
        return self.fc(lstm_out)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, train_batch, batch_idx)
        self.log("train_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = shared_eval_step(self, val_batch, batch_idx)
        self.log("val_loss", loss)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, training_step_outputs)

        self.log("train_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        scaled_loss = shared_eval_epoch_end(self, validation_step_outputs)

        self.log("val_rmse_scaled_epoch", scaled_loss,
                 on_epoch=True, on_step=False)
