import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.data.batch import Batch
from base_models import HeteroGLSTM
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import mean_squared_error, r2_score

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

        self.model = HeteroGLSTM(cfg.model.num_layers,
                                 cfg.model.hidden_channels,
                                 cfg.model.n_outputs, metadata)
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

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        out = self(batch.x_dict, batch.edge_index_dict)
        out = out.squeeze()

        loss = self.loss_fn(batch.y_dict["measurement"], out)

        self.log("train_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return loss

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        out, loss = self._shared_eval_step(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return {'loss': loss, 'outputs': out,
                'targets': batch.y_dict["measurement"]}

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        out, loss = self._shared_eval_step(batch, batch_idx)

        self.log("test_loss", loss,
                 batch_size=self.cfg.training.batch_size)

        return loss

    def validation_epoch_end(
        self, validation_step_outputs: List[Dict[str, torch.Tensor]]
    ) -> None:

        outputs, targets = [], []

        for out in validation_step_outputs:
            outputs.append(out["outputs"])
            targets.append(out["targets"])

        outputs = torch.cat(outputs).cpu().numpy().reshape(-1, 1)
        targets = torch.cat(targets).cpu().numpy().reshape(-1, 1)

        outputs = self.scaler.inverse_transform(outputs)
        targets = self.scaler.inverse_transform(targets)

        r2 = r2_score(targets, outputs)
        scaled_loss = mean_squared_error(targets, outputs)

        self.log("val_loss_scaled", scaled_loss, on_epoch=True, on_step=False)
        self.log("r2_scaled", r2, on_epoch=True, on_step=False)

    def _shared_eval_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out = self(batch.x_dict, batch.edge_index_dict)
        out = out.squeeze()
        loss = self.loss_fn(batch.y_dict["measurement"], out)

        return out, loss


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]
