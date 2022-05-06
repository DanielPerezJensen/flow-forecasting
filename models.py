import torch
import pytorch_lightning as pl
from torch import nn
from base_models import HeteroGLSTM
from omegaconf import DictConfig, OmegaConf

from typing import Any, Optional, Tuple, List
# Typing options
Opt_Dict = Optional[dict]


class HeteroGLSTM_pl(pl.LightningModule):
    def __init__(
                self,
                cfg: DictConfig,
                metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                optimizer: torch.optim.Optimizer
            ) -> None:
        super().__init__()

        self.model = HeteroGLSTM(cfg.model.num_layers,
                                 cfg.model.hidden_channels,
                                 cfg.model.n_outputs, metadata)
        self.linear = nn.Linear(cfg.model.hidden_channels, cfg.model.n_outputs)

        self.activation = nn.ReLU()
        self.loss_fn = nn.MSELoss()

        self.optimizer = optimizer

        # Store some hyperparameters
        self.cfg = cfg

    def forward(self, x_dict: dict, edge_index_dict: dict,
                h: Opt_Dict = None, c: Opt_Dict = None) -> torch.Tensor:

        h, c = self.model(x_dict, edge_index_dict, h_dict=h, c_dict=c)
        out = self.activation(h["measurement"])

        out = self.linear(out)

        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), self.cfg.optimizer.lr)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        out = self(batch.x_dict, batch.edge_index_dict)
        out = out.squeeze()

        loss = self.loss_fn(batch.y_dict["measurement"], out)

        self.log("train_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, on_step=True,
                 batch_size=self.cfg.training.batch_size)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)

        self.log("test_loss", loss,
                 batch_size=self.cfg.training.batch_size)

        return loss

    def _shared_eval_step(self, batch, batch_idx) -> torch.Tensor:
        out = self(batch.x_dict, batch.edge_index_dict)
        out = out.squeeze()

        loss = self.loss_fn(batch.y_dict["measurement"], out)

        return loss
