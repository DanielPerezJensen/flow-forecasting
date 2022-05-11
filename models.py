import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.data.batch import Batch
from base_models import HeteroGLSTM
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from typing import Any, Optional, Tuple, List, Dict, Callable, Type

# Typing options
TensorDict = Dict[str, torch.Tensor]


class HeteroGLSTM_pl(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: Type
    ) -> None:

        super().__init__()

        self.metadata = metadata

        self.model = HeteroGLSTM(cfg.model.num_layers,
                                 cfg.model.hidden_channels,
                                 cfg.model.n_outputs, self.metadata)
        self.linear = nn.Linear(cfg.model.hidden_channels, cfg.model.n_outputs)

        self.activation = nn.ReLU()
        self.loss_fn = nn.MSELoss()

        self.scaler = scaler

        # Store some hyperparameters
        self.cfg = cfg

    def forward(
        self, x_dict: TensorDict, edge_index_dict: TensorDict,
        h_dict: Optional[TensorDict] = None,
        c_dict: Optional[TensorDict] = None
    ) -> TensorDict:

        h, c = self.model(x_dict, edge_index_dict,
                          h_dict=h_dict, c_dict=c_dict)
        out = self.activation(h["measurement"])

        out = self.linear(out)

        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Extract optimizer
        self.optimizer = get_optimizer(self.cfg.optimizer.name)
        return self.optimizer(self.parameters(), self.cfg.optimizer.lr)

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

    def test_epoch_end(
        self, test_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        rmse, r2 = self._shared_eval_epoch(test_step_outputs)

        self.log("test_rmse_scaled_epoch", rmse, on_step=False, on_epoch=True)
        self.log("test_r2_scaled_epoch", r2, on_step=False, on_epoch=True)

    def training_epoch_end(
        self, training_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        rmse, r2 = self._shared_eval_epoch(training_step_outputs)

        self.log("train_rmse_scaled_epoch", rmse, on_step=False, on_epoch=True)
        self.log("train_r2_scaled_epoch", r2, on_step=False, on_epoch=True)

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        rmse, r2 = self._shared_eval_epoch(validation_step_outputs)

        self.log("val_rmse_scaled_epoch", rmse, on_step=False, on_epoch=True)
        self.log("val_r2_scaled_epoch", r2, on_step=False, on_epoch=True)

    def _shared_eval_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        output = self(batch.x_dict, batch.edge_index_dict)
        target = batch.y_dict["measurement"]

        n_target_stations = len(self.cfg.data.target_stations)

        # Output will contain predicted outputs for all stations,
        # but we only want the ones contained in the configuration
        output = output.reshape(-1, 4)
        output = output[:, self.cfg.data.target_stations]

        # Targets only contain target stations,
        # so only reshape to amount of target stations
        target = target.reshape(-1, n_target_stations)

        loss = self.loss_fn(target, output)

        return loss, output, target

    def _shared_eval_epoch(
        self, step_outputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[float, float]:

        outputs_list, targets_list = [], []

        for out in step_outputs:
            outputs_list.append(out["outputs"])
            targets_list.append(out["targets"])

        n_target_stations = len(self.cfg.data.target_stations)

        np_outputs = torch.cat(outputs_list).detach().cpu().numpy()
        np_targets = torch.cat(targets_list).detach().cpu().numpy()

        descaled_outputs = self.scaler.inverse_transform(np_outputs)
        descaled_targets = self.scaler.inverse_transform(np_targets)

        rmse = mean_squared_error(descaled_targets, descaled_outputs,
                                  squared=False)
        r2 = r2_score(descaled_targets, descaled_outputs)

        return rmse, r2


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]
