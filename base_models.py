import torch
import torch.nn as nn
import data
import os
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv, Linear
from torch_geometric.loader import DataLoader

from typing import Optional, Tuple, Any


class Gate(nn.Module):
    def __init__(self, out_channels: int, num_layers: int,
                 metadata: tuple, bias=True):
        super().__init__()

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=out_channels,
                                    bias=bias)
                for edge_type in metadata[1]
            })

            self.convs.append(conv)

        self.linear = nn.ModuleDict({
                        node_type: Linear(-1, out_channels, bias=bias) for node_type in metadata[0]
                    })

        self.activation = nn.Sigmoid()

    def forward(self, x_dict: dict, edge_index_dict: dict, h_dict: dict) -> dict:
        for conv in self.convs:
            out_dict = conv(x_dict, edge_index_dict)

        for node_type, x in out_dict.items():
            out_dict[node_type] = self.linear[node_type](x)
            out_dict[node_type] = self.activation(out_dict[node_type])

        return out_dict


class CellGate(nn.Module):
    def __init__(self, out_channels: int, num_layers: int,
                 metadata: tuple, bias=True) -> None:
        super().__init__()

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=out_channels,
                                    bias=bias)
                for edge_type in metadata[1]
            })

            self.convs.append(conv)

        self.linear = nn.ModuleDict({
                        node_type: Linear(-1, out_channels, bias=bias) for node_type in metadata[0]
                    })

        self.activation = nn.Tanh()

    def forward(self, x_dict: dict, edge_index_dict: dict,
                h_dict: dict, c_dict: dict, i_dict: dict, f_dict: dict) -> None:

        for conv in self.convs:
            t_dict = conv(x_dict, edge_index_dict)

        for node_type, x in t_dict.items():
            t_dict[node_type] = self.linear[node_type](x)
            t_dict[node_type] = self.activation(t_dict[node_type])

        out_dict = {}

        for node_type, C in c_dict.items():
            out_dict[node_type] = f_dict[node_type] * C + i_dict[node_type] * t_dict[node_type]

        return out_dict


class HeteroGLSTM(nn.Module):
    def __init__(
            self,
            num_layers: int,
            hidden_channels: int,
            out_channels: int,
            metadata: tuple,
            bias: bool = True
    ) -> None:

        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.bias = bias

        self.i_gate = Gate(hidden_channels, num_layers, metadata, bias=bias)
        self.f_gate = Gate(hidden_channels, num_layers, metadata, bias=bias)
        self.o_gate = Gate(hidden_channels, num_layers, metadata, bias=bias)

        self.c_gate = CellGate(hidden_channels, num_layers, metadata, bias=bias)

    def _set_hidden_state(self, x_dict: dict, h_dict: Optional[dict]) -> dict:
        if h_dict is None:
            h_dict = nn.ParameterDict()
            for node_type, X in x_dict.items():
                h_dict[node_type] = nn.Parameter(torch.zeros(X.shape[0], self.hidden_channels).to(X.device))

        return h_dict

    def _set_cell_state(self, x_dict: dict, c_dict: Optional[dict]) -> dict:
        if c_dict is None:
            c_dict = nn.ParameterDict()
            for node_type, X in x_dict.items():
                c_dict[node_type] = nn.Parameter(torch.zeros(X.shape[0], self.hidden_channels).to(X.device))

        return c_dict

    def _calculate_hidden_state(self, o_dict: dict, c_dict: dict) -> dict:
        h_dict = {}
        for node_type in o_dict.keys():
            h_dict[node_type] = o_dict[node_type] * torch.tanh(c_dict[node_type])

        return h_dict

    def forward(
                self,
                x_dict: dict,
                edge_index_dict: dict,
                h_dict: Optional[dict] = None,
                c_dict: Optional[dict] = None
            ) -> Tuple[dict, dict]:

        h_dict = self._set_hidden_state(x_dict, h_dict)
        c_dict = self._set_cell_state(x_dict, c_dict)

        i_dict = self.i_gate(x_dict, edge_index_dict, h_dict)
        f_dict = self.f_gate(x_dict, edge_index_dict, h_dict)

        c_dict = self.c_gate(x_dict, edge_index_dict, h_dict, c_dict, i_dict, f_dict)

        o_dict = self.o_gate(x_dict, edge_index_dict, h_dict)

        h_dict = self._calculate_hidden_state(o_dict, c_dict)

        return h_dict, c_dict


if __name__ == "__main__":

    freq = "W"
    lag = 24

    processed_path = os.path.join("data", "processed")
    dataset = data.GraphRiverFlowDataset(processed_path, freq=freq, lag=lag,
                                         scaler_name="maxabs", process=True)

    train, val, test = data.split_dataset(dataset, freq=freq, lag=lag)
    train_loader = DataLoader(train, batch_size=1)
    val_loader = DataLoader(val, batch_size=1)
    test_loader = DataLoader(test, batch_size=1)

    device = torch.device("cpu")

    model = HeteroGLSTM(3, 64, 4, dataset[0].metadata(), bias=True)

    # Dummy iteration for loading of lazy layer sizes
    with torch.no_grad():
        h_dict, c_dict = model(dataset[0].x_dict, dataset[0].edge_index_dict)

    print(model)

    model = model.to(device)

    for batch in train_loader:
        batch = batch.to(device)
        h_dict, c_dict = model(batch.x_dict, batch.edge_index_dict)
        print(h_dict["measurement"])

        break
