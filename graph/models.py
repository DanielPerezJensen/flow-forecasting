import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.data.batch import Batch
from base_models import HeteroGLSTM
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


class HeteroGLSTM_pl(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        scaler: ScalerType
    ) -> None:

        super().__init__()

        self.model = HeteroGLSTM(cfg.model.num_layers, cfg.model.out_channels,
                                 metadata)
        self.linear = nn.Linear(cfg.model.out_channels, cfg.model.n_outputs)

        self.activation = nn.ReLU()
        self.loss_fn = nn.MSELoss()

        self.scaler = scaler

        # Store some hyperparameters
        self.cfg = cfg

    def forward(
        self, x_dict: TensorDict, edge_index_dict: TensorDict,
        h_dict: Optional[TensorDict] = None,
        c_dict: Optional[TensorDict] = None
    ) -> Any:

        h, c = self.model(x_dict, edge_index_dict,
                          h_dict=h_dict, c_dict=c_dict)
        out = self.linear(self.activation(h["measurement"]))

        return out

    def configure_optimizers(
        self
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optimizer = get_optimizer(self.cfg.optimizer.name)
        self.optimizer = optimizer(self.parameters(), **self.cfg.optimizer.hparams)

        return self.optimizer

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

        self.log_dict(eval_dict, on_epoch=True)

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
    ) -> Dict[str, float]:

        outputs_list, targets_list = [], []

        for out in step_outputs:
            outputs_list.append(out["outputs"])
            targets_list.append(out["targets"])

        n_target_stations = len(self.cfg.data.target_stations)

        np_outputs = torch.cat(outputs_list).detach().cpu().numpy()
        np_targets = torch.cat(targets_list).detach().cpu().numpy()

        descaled_outputs = self.scaler.inverse_transform(np_outputs)
        descaled_targets = self.scaler.inverse_transform(np_targets)

        eval_dict = get_evaluation_measures(descaled_outputs, descaled_targets)

        return eval_dict


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
    kge_outs = []
    kge_prime_outs = []

    for p, t in zip(predictions.T, targets.T):
        nse_outs.append(he.evaluator(he.nse, p, t)[0])
        kge_outs.append(he.evaluator(he.kge, p, t)[0])
        kge_prime_outs.append(he.evaluator(he.kgeprime, p, t)[0])

    nse_mean = np.average(nse_outs)
    kge_mean = np.average(kge_outs)
    kge_prime_mean = np.average(kge_prime_outs)

    eval_dict = {
        "rmse": mean_squared_error(targets, predictions, squared=False),
        "r2_score": r2_score(targets, predictions),
        "nse": nse_mean,
        "nnse": 1 / (2 - nse_mean),
        "kge": kge_mean,
        "kgeprime": kge_prime_mean
    }

    return eval_dict


def get_optimizer(optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:
    """
    function: get_optimizer

    Returns an optimizer given it's name
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]
