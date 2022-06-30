import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric import nn as geom_nn
from torch_geometric.data.batch import Batch
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import mean_squared_error, r2_score
import hydroeval as he
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from typing import Any, Optional, Tuple, List, Dict, Callable, Type, Union
import numpy.typing as npt

# Typing options
TensorDict = Dict[str, torch.Tensor]
ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, metadata, scaler):
        super().__init__()

        self.cfg = cfg
        self.metadata = metadata
        self.scaler = scaler


    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, output, target = self._shared_eval_step(batch, batch_idx)

        self.log("train_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return {"loss": loss, "outputs": output, "targets": target}

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, output, target = self._shared_eval_step(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return {"loss": loss, "outputs": output, "targets": target}

    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, output, target = self._shared_eval_step(batch, batch_idx)

        return {"loss": loss, "outputs": output, "targets": target}

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        eval_dict = self._shared_eval_epoch(training_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"train_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True, prog_bar=True)

    def validation_epoch_end(
        self, validation_step_outputs
    ) -> None:

        eval_dict = self._shared_eval_epoch(validation_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"val_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True, prog_bar=True)

    def test_epoch_end(
        self, test_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        eval_dict = self._shared_eval_epoch(test_step_outputs)

        for k in eval_dict.copy():
            eval_dict[f"test_{k}"] = eval_dict.pop(k)

        self.log_dict(eval_dict, on_epoch=True)

    def _shared_eval_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.cfg.data.sequential:
            output = self(batch.xs_dict, batch.edge_indices_dict)
        else:
            output = self(batch.x_dict, batch.edge_index_dict)

        target = batch.y_dict["measurement"]

        loss = self.loss_fn(output, target)

        return loss, output, target

    def _shared_eval_epoch(
        self, step_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:

        outputs_list, targets_list = [], []

        for out in step_outputs:
            outputs_list.append(out["outputs"].reshape(-1, 4))
            targets_list.append(out["targets"].reshape(-1, 4))

        np_outputs = torch.cat(outputs_list).detach().cpu().numpy()
        np_targets = torch.cat(targets_list).detach().cpu().numpy()

        descaled_outputs = self.scaler.inverse_transform(np_outputs)
        descaled_targets = self.scaler.inverse_transform(np_targets)

        eval_dict = get_evaluation_measures(descaled_outputs, descaled_targets)

        return eval_dict


class HeteroMLP(BaseModel):
    def __init__(
        self,
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: ScalerType,
    ) -> None:

        super().__init__(cfg, metadata, scaler)

        self.cfg = cfg
        self.metadata = metadata
        self.scaler = scaler

        self.convs = nn.ModuleList()

        # Convolutions dictated by config
        if self.cfg.model.convolution.name == "sage":
            conv_out_dim = cfg.model.convolution.out_channels

            for _ in range(cfg.model.convolution.num_layers):
                conv = geom_nn.HeteroConv({
                    edge_type: geom_nn.SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=cfg.model.convolution.out_channels
                    )
                    for edge_type in metadata[1]
                })

                self.convs.append(conv)

        elif self.cfg.model.convolution.name == "gat":

            conv_out_dim = (cfg.model.convolution.out_channels *
                            cfg.model.convolution.n_heads)

            for _ in range(cfg.model.convolution.num_layers):
                conv = geom_nn.HeteroConv({
                    edge_type: geom_nn.GATv2Conv(
                        in_channels=(-1, -1),
                        out_channels=cfg.model.convolution.out_channels,
                        heads=cfg.model.convolution.heads,
                        dropout=cfg.model.convolution.dropout_prob,
                    )
                    for edge_type in metadata[1]
                })

                self.convs.append(conv)

        # Layer Norm for each node type
        self.conv_normalizations = nn.ModuleDict(
            {node: geom_nn.LayerNorm(conv_out_dim)
                for node in metadata[0]}
        )

        # The lag is dictated by the data frequency
        if cfg.data.freq == "M":
            self.lag = 6
        elif cfg.data.freq == "W":
            self.lag = 24

        self.mlp = geom_nn.MLP(
            in_channels=cfg.model.convolution.out_channels,
            hidden_channels=cfg.model.hidden_dim,
            out_channels=6, num_layers=cfg.model.mlp_layers,
            dropout=cfg.model.mlp_dropout_prob,
            batch_norm=False
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x_dict, edge_index_dict):
        # Apply message passing
        for conv in self.convs:
            conv_out = conv(x_dict, edge_index_dict)

        # Normalize output of message passing
        for node_type, out in conv_out.items():
            conv_out[node_type] = self.conv_normalizations[node_type](out)

        # Take out measurement nodes and reshape to proper output
        msr_out = conv_out["measurement"]

        mlp_out = self.mlp(msr_out)

        return mlp_out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        self.optimizer = optimizer(self.parameters(),
                                   **self.cfg.optimizer.hparams)

        return self.optimizer


class HeteroSeqGRU(BaseModel):
    def __init__(
        self,
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: ScalerType,
    ) -> None:

        super().__init__(cfg, metadata, scaler)

        self.cfg = cfg
        self.metadata = metadata
        self.scaler = scaler

        self.convs = nn.ModuleList()

        # Convolutions dictated by config
        if self.cfg.model.convolution.name == "sage":
            conv_out_dim = cfg.model.convolution.out_channels

            for _ in range(cfg.model.convolution.num_layers):
                conv = geom_nn.HeteroConv({
                    edge_type: geom_nn.SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=cfg.model.convolution.out_channels
                    )
                    for edge_type in metadata[1]
                })

                self.convs.append(conv)

        elif self.cfg.model.convolution.name == "gat":

            conv_out_dim = (cfg.model.convolution.out_channels *
                            cfg.model.convolution.n_heads)

            for _ in range(cfg.model.convolution.num_layers):
                conv = geom_nn.HeteroConv({
                    edge_type: geom_nn.GATv2Conv(
                        in_channels=(-1, -1),
                        out_channels=cfg.model.convolution.out_channels,
                        heads=cfg.model.convolution.heads,
                        dropout=cfg.model.convolution.dropout_prob,
                    )
                    for edge_type in metadata[1]
                })

                self.convs.append(conv)

        # Layer Norm for each node type
        self.conv_normalizations = nn.ModuleDict(
            {node: geom_nn.LayerNorm(conv_out_dim)
                for node in metadata[0]}
        )

        # The lag is dictated by the data frequency
        if cfg.data.freq == "M":
            self.lag = 6
        elif cfg.data.freq == "W":
            self.lag = 24

        self.gru = nn.GRU(
            cfg.model.convolution.out_channels, cfg.model.hidden_dim,
            batch_first=True
        )

        # Layer norm for output of GRU
        self.layer_norm = nn.LayerNorm(cfg.model.hidden_dim)

        self.linear = nn.Linear(cfg.model.hidden_dim, 6)

        self.loss_fn = nn.MSELoss()

    def forward(
        self, x_dict: TensorDict, edge_index_dict: TensorDict
    ) -> torch.Tensor:
        batch_size = x_dict["measurement"].size(0)

        # [B, T, N, D] -> [B x T x N, D]
        for node_type in x_dict:
            x_dict[node_type] = torch.flatten(x_dict[node_type], 0, 2)

        # Format edges to expected format of -> [2, M]
        for edge_type in edge_index_dict:
            edge_index_dict[edge_type] = edge_index_dict[
                edge_type
            ].permute((0, 2, 1)).flatten(0, 1).T

        # Apply message passing
        for conv in self.convs:
            conv_out = conv(x_dict, edge_index_dict)

        # Normalize output of message passing
        for node_type, out in conv_out.items():
            conv_out[node_type] = self.conv_normalizations[node_type](out)

        # Take out measurement nodes and reshape to proper output
        msr_out = conv_out["measurement"]

        # Reshape to [batch_size, lag, n_stations, dimension]
        msr_out = msr_out.reshape((batch_size, self.lag, 4, -1))
        # Reshape to [batch_size, n_stations, lag, dimension]
        msr_out = msr_out.permute((0, 2, 1, 3))
        # Flatten into [batch_size * n_stations, lag, dimension]
        msr_out = msr_out.flatten(0, 1)

        gru_out, _ = self.gru(msr_out)

        # Normalize output of GRU
        gru_out = self.layer_norm(gru_out)

        gru_out = gru_out[:, -1, :]
        gru_out = gru_out.reshape((batch_size, 4, -1))
        lin_out = self.linear(gru_out)  # type: torch.Tensor

        return lin_out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        self.optimizer = optimizer(self.parameters(),
                                   **self.cfg.optimizer.hparams)

        return self.optimizer


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
        "rmse": mean_squared_error(predictions, targets, squared=False),
        "nse": nse_mean,
    }

    return eval_dict


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:
    """
    function: get_optimizer

    Returns an optimizer given it's name
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
    }

    return optimizer_dict[optimizer_name]
