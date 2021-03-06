"""
This file contains various models used throughout the project
"""
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score
from omegaconf import DictConfig, OmegaConf

import hydroeval as he
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from torch import Tensor
from typing import Any, Optional, Tuple, List, Dict, Callable, Type, Union
import numpy.typing as npt


ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]


def get_evaluation_measures(
    predictions: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32]
) -> Dict[str, float]:
    """
    function:

    Returns a dictionary of evaluation measures according to provided
    predictions and targets
    """
    nse_outs = []

    for p, t in zip(predictions.T, targets.T):
        nse_outs.append(he.evaluator(he.nse, p, t)[0])

    nse_mean = np.average(nse_outs)

    eval_dict = {
        "rmse": mean_squared_error(targets, predictions, squared=False),
        "nse": nse_mean,
    }

    return eval_dict


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]


class BaseModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = self._shared_eval_step(train_batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=True)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = self._shared_eval_step(val_batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, on_step=True)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def test_step(
        self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        loss, outputs, targets = self._shared_eval_step(test_batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=True)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        eval_dict = self._shared_eval_epoch_end(training_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"train_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True, prog_bar=True)

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        eval_dict = self._shared_eval_epoch_end(validation_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"val_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True, prog_bar=True)

    def test_epoch_end(
        self, test_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        eval_dict = self._shared_eval_epoch_end(test_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"test_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True)

    def _shared_eval_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs, targets = batch

        outputs = self(inputs)

        outputs = outputs.view(
            -1, len(self.cfg.data.target_stations), self.output_dim
        )

        loss = self.loss_fn(outputs, targets)

        return loss, outputs, targets

    def _shared_eval_epoch_end(
        self, step_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:

        outputs_list, targets_list = [], []

        for out in step_outputs:
            outputs_list.append(out["outputs"].reshape(-1, len(self.cfg.data.target_stations)))
            targets_list.append(out["targets"].reshape(-1, len(self.cfg.data.target_stations)))

        np_outputs = torch.cat(outputs_list).detach().cpu().numpy()
        np_targets = torch.cat(targets_list).detach().cpu().numpy()

        descaled_outputs = self.scaler.inverse_transform(np_outputs)
        descaled_targets = self.scaler.inverse_transform(np_targets)

        eval_dict = get_evaluation_measures(descaled_outputs, descaled_targets)

        return eval_dict


class MLP(BaseModel):
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

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = []  # type: List[nn.Module]
        layer_sizes = [input_dim] + list(self.cfg.model.layers)

        for layer_index in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[layer_index - 1],
                                         layer_sizes[layer_index]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(
            layer_sizes[-1], len(cfg.data.target_stations) * output_dim
        ))

        # Typing of the sequential layers
        self.model: Callable[[torch.Tensor], torch.Tensor]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)


class GRU(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scaler: ScalerType,
        cfg: DictConfig
    ) -> None:
        super().__init__()

        self.cfg = cfg
        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = nn.MSELoss()

        # GRU/LSTM layers
        self.model = nn.GRU(
            self.input_dim, cfg.model.hidden_dim, cfg.model.layer_dim,
            batch_first=True, dropout=cfg.model.dropout_prob
        )

        self.layer_norm = nn.LayerNorm(cfg.model.hidden_dim)

        # Typing option of fully connected output layer
        self.fc: Callable[[torch.Tensor], torch.Tensor]
        self.fc = nn.Linear(
            cfg.model.hidden_dim, len(cfg.data.target_stations) * self.output_dim
        )

        self.scaler = scaler
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(
            self.cfg.model.layer_dim, x.size(0),
            self.cfg.model.hidden_dim, device=x.device
        ).requires_grad_().to(x.device)

        # Forward propagation by passing in the input and hidden state
        gru_out, _ = self.model(x, h0.detach())

        # Reshaping the outputs in the shape of
        # (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        gru_out = gru_out[:, -1, :]
        gru_out = self.layer_norm(gru_out)

        # Convert the final state to our desired output shape
        # (batch_size, output_dim)
        return self.fc(gru_out)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)


class LSTM(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scaler: ScalerType,
        cfg: DictConfig
    ) -> None:

        super().__init__()

        self.cfg = cfg
        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = nn.MSELoss()

        # GRU/LSTM layers
        self.model = nn.LSTM(
            input_dim, cfg.model.hidden_dim, cfg.model.layer_dim,
            batch_first=True, dropout=cfg.model.dropout_prob
        )

        self.layer_norm = nn.LayerNorm(cfg.model.hidden_dim)

        # Typing option of fully connected output layer
        self.fc: Callable[[torch.Tensor], torch.Tensor]
        self.fc = nn.Linear(
            cfg.model.hidden_dim, len(cfg.data.target_stations) * output_dim
        )

        self.scaler = scaler
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(
            self.cfg.model.layer_dim, x.size(0), self.cfg.model.hidden_dim
        ).requires_grad_().to(x.device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(
            self.cfg.model.layer_dim, x.size(0), self.cfg.model.hidden_dim
        ).requires_grad_().to(x.device)

        lstm_out, (hn, cn) = self.model(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of
        # so that it can fit into the fully connected layer
        lstm_out = lstm_out[:, -1, :]

        lstm_out = self.layer_norm(lstm_out)

        return self.fc(lstm_out)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.hparams)
