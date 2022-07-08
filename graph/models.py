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


class MLPConv(geom_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels[0] + in_channels[1], out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.lin_l = nn.Linear(out_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(out_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        out = self.propagate(edge_index, x=x)

        out = self.lin_l(out)

        return out

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, edge_channels]
        # tmp has shape [E, 2 * in_channels + edge_channels]
        tmp = torch.cat([x_i, x_j], dim=1)

        return self.mlp(tmp)


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
        in_channels_dict: Dict[str, int],
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: ScalerType,
    ) -> None:

        super().__init__(cfg, metadata, scaler)

        self.lin_dict = nn.ModuleDict()

        # Node transformations to out_channels
        for node_type in metadata[0]:
            self.lin_dict[node_type] = geom_nn.Linear(
                -1, cfg.model.convolution.out_channels
            )

        self.convs = get_convs(metadata, **cfg.model.convolution)

        conv_out_dim = cfg.model.convolution.out_channels

        if cfg.model.convolution.name == "gat":
            conv_out_dim = cfg.model.convolution.out_channels * cfg.model.convolution.heads

        self.mlp = geom_nn.MLP(
            in_channels=conv_out_dim,
            hidden_channels=cfg.model.hidden_dim,
            out_channels=6, num_layers=cfg.model.mlp_layers,
            dropout=cfg.model.mlp_dropout_prob,
            batch_norm=False
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x_dict, edge_index_dict):

        x_dict = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # Take out measurement nodes and reshape to proper output
        msr_out = x_dict["measurement"]

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
        in_channels_dict: Dict[str, int],
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: ScalerType,
    ) -> None:

        super().__init__(cfg, metadata, scaler)

        self.lin_dict = nn.ModuleDict()

        # Node transformations to out_channels
        for node_type in metadata[0]:
            self.lin_dict[node_type] = geom_nn.Linear(
                -1, cfg.model.convolution.out_channels 
            )

        self.convs = get_convs(metadata, **cfg.model.convolution)

        # The lag is dictated by the data frequency
        if cfg.data.freq == "M":
            self.lag = 6
        elif cfg.data.freq == "W":
            self.lag = 24

        conv_out_dim = cfg.model.convolution.out_channels

        if cfg.model.convolution.name == "gat":
            conv_out_dim = cfg.model.convolution.out_channels * cfg.model.convolution.heads

        self.rnn = nn.LSTM(
            conv_out_dim, cfg.model.hidden_dim,
            batch_first=True
        )

        self.linear = nn.Linear(cfg.model.hidden_dim, 6)

        self.loss_fn = nn.MSELoss()

    def forward(
        self, x_dict: TensorDict, edge_index_dict: TensorDict
    ) -> torch.Tensor:
        batch_size = x_dict["measurement"].size(0)

        # [B, T, N, D] -> [B x T x N, D] -> [B x T x N, H]
        x_dict = {
            node_type: self.lin_dict[node_type](
                torch.flatten(x_dict[node_type], 0, 2)
            ) for node_type, x in x_dict.items()
        }

        # Format edges to expected format of -> [2, M]
        for edge_type in edge_index_dict:
            edge_index_dict[edge_type] = edge_index_dict[
                edge_type
            ].permute((0, 2, 1)).flatten(0, 1).T

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # Take out measurement nodes and reshape to proper output
        msr_out = x_dict["measurement"]

        # Reshape to [batch_size, lag, n_stations, dimension]
        msr_out = msr_out.reshape((batch_size, self.lag, 4, -1))
        # Reshape to [batch_size, n_stations, lag, dimension]
        msr_out = msr_out.permute((0, 2, 1, 3))
        # Flatten into [batch_size * n_stations, lag, dimension]
        msr_out = msr_out.flatten(0, 1)
        gru_out, _ = self.rnn(msr_out)

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


def get_convs(
    metadata: Tuple[List[str], List[Tuple[str, str, str]]],
    **kwargs: Any
) -> nn.ModuleList:

    # Convolutions dictated by config
    convs = nn.ModuleList()

    if kwargs["name"] == "sage":
        for _ in range(kwargs["num_layers"]):
            convs.append(geom_nn.HeteroConv({
                edge_type: geom_nn.SAGEConv(
                    in_channels=-1,
                    out_channels=kwargs["out_channels"],
                ) for edge_type in metadata[1]
            }))

    elif kwargs["name"] == "gat":
        for _ in range(kwargs["num_layers"]):
            convs.append(geom_nn.HeteroConv({
                edge_type: geom_nn.GATv2Conv(
                    in_channels=-1,
                    out_channels=kwargs["out_channels"],
                    heads=kwargs["heads"],
                    dropout=kwargs["dropout_prob"],
                ) for edge_type in metadata[1]
            }))

    elif kwargs["name"] == "han":
        for _ in range(kwargs["num_layers"]):
            convs.append(geom_nn.HANConv(
                in_channels=kwargs["out_channels"],
                out_channels=kwargs["out_channels"],
                metadata=metadata,
                heads=kwargs["heads"],
                dropout=kwargs["dropout_prob"],
            ))

    return convs
