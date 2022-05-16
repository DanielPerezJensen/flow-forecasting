import torch
import torch.nn as nn
import data
import os
import torch_geometric.nn as geom_nn
from torch_geometric.loader import DataLoader
from torch.nn.parameter import Parameter

from typing import Optional, Tuple, Any, Dict, List

# Typing options
TensorDict = Dict[str, torch.Tensor]
MetadataType = Tuple[List[str], List[Tuple[str, str, str]]]


class Gate(nn.Module):
    def __init__(
        self, out_channels: int, num_layers: int,
        metadata: MetadataType, bias: bool = True
    ) -> None:
        super().__init__()

        # Initialize linear weights
        self.linears = nn.ModuleDict({
                        node_type: geom_nn.Linear(-1, out_channels, bias=False)
                        for node_type in metadata[0]
                    })

        # Initialize bias weights
        self.biases = nn.ParameterDict({
                node_type: Parameter(torch.empty(out_channels))
                for node_type in metadata[0]
            })

        for node_type in self.biases:
            nn.init.uniform_(self.biases[node_type])

        # Set up convolution layers
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = geom_nn.HeteroConv({
                edge_type: geom_nn.SAGEConv(in_channels=(-1, -1),
                                            out_channels=out_channels,
                                            bias=bias)
                for edge_type in metadata[1]
            })

            self.convs.append(conv)

        self.activation = nn.Sigmoid()

    def forward(
        self, x_dict: TensorDict, edge_index_dict: TensorDict,
        h_dict: TensorDict
    ) -> TensorDict:
        out_dict = {node_type: self.linears[node_type](X)
                    for node_type, X in x_dict.items()}

        for conv in self.convs:
            out_dict = conv(out_dict, edge_index_dict)

        out_dict = {node_type: self.activation(self.biases[node_type] + X)
                    for node_type, X in out_dict.items()}

        return out_dict


class CellGate(nn.Module):
    def __init__(
        self, out_channels: int, num_layers: int,
        metadata: MetadataType, bias: bool = True
    ) -> None:

        super().__init__()

        # Initialize linear weights
        self.linears = nn.ModuleDict({
                        node_type: geom_nn.Linear(-1, out_channels, bias=False)
                        for node_type in metadata[0]
                    })

        # Initialize bias weights
        self.biases = nn.ParameterDict({
                node_type: Parameter(torch.empty(out_channels))
                for node_type in metadata[0]
            })

        for node_type in self.biases:
            nn.init.uniform_(self.biases[node_type])

        # Set up convolution layers
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = geom_nn.HeteroConv({
                edge_type: geom_nn.SAGEConv(in_channels=(-1, -1),
                                            out_channels=out_channels,
                                            bias=bias)
                for edge_type in metadata[1]
            })

            self.convs.append(conv)

        self.activation = nn.Tanh()

    def forward(
        self,
        x_dict: TensorDict, edge_index_dict: TensorDict,
        h_dict: TensorDict, c_dict: TensorDict,
        i_dict: TensorDict, f_dict: TensorDict
    ) -> TensorDict:
        t_dict = {node_type: self.linears[node_type](X)
                  for node_type, X in x_dict.items()}

        for conv in self.convs:
            t_dict = conv(t_dict, edge_index_dict)

        t_dict = {node_type: self.activation(self.biases[node_type] + X)
                  for node_type, X in t_dict.items()}

        out_dict = {node_type: f_dict[node_type] * C +
                    i_dict[node_type] * t_dict[node_type]
                    for node_type, C in c_dict.items()}

        return out_dict


class HeteroGLSTM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        out_channels: int,
        metadata: MetadataType,
        bias: bool = True
    ) -> None:

        super().__init__()

        self.num_layers = num_layers
        self.out_channels = out_channels
        self.metadata = metadata
        self.bias = bias

        self.i_gate = Gate(out_channels, num_layers, metadata, bias=bias)
        self.f_gate = Gate(out_channels, num_layers, metadata, bias=bias)
        self.o_gate = Gate(out_channels, num_layers, metadata, bias=bias)

        self.c_gate = CellGate(out_channels, num_layers,
                               metadata, bias=bias)

    def _set_hidden_state(
        self, x_dict: TensorDict, h_dict: Optional[nn.ParameterDict]
    ) -> nn.ParameterDict:

        if h_dict is None:
            h_dict = nn.ParameterDict()
            for node_type, X in x_dict.items():
                h_dict[node_type] = Parameter(
                    torch.zeros(X.shape[0], self.out_channels).to(X.device)
                )

        return h_dict

    def _set_cell_state(
        self, x_dict: TensorDict, c_dict: Optional[nn.ParameterDict]
    ) -> nn.ParameterDict:

        if c_dict is None:
            c_dict = nn.ParameterDict()
            for node_type, X in x_dict.items():
                c_dict[node_type] = Parameter(
                    torch.zeros(X.shape[0], self.out_channels).to(X.device)
                )

        return c_dict

    def _calculate_hidden_state(
        self, o_dict: TensorDict, c_dict: nn.ParameterDict
    ) -> TensorDict:

        h_dict = {}
        for node_type in o_dict.keys():
            h_dict[node_type] = (o_dict[node_type] *
                                 torch.tanh(c_dict[node_type]))

        return h_dict

    def forward(
        self,
        x_dict: TensorDict,
        edge_index_dict: TensorDict,
        h_dict: Optional[nn.ParameterDict] = None,
        c_dict: Optional[nn.ParameterDict] = None
    ) -> Tuple[TensorDict, TensorDict]:

        h_dict = self._set_hidden_state(x_dict, h_dict)
        c_dict = self._set_cell_state(x_dict, c_dict)

        i_dict = self.i_gate(x_dict, edge_index_dict, h_dict)
        f_dict = self.f_gate(x_dict, edge_index_dict, h_dict)

        c_dict_new = self.c_gate(x_dict, edge_index_dict, h_dict,
                                 c_dict, i_dict, f_dict)

        o_dict = self.o_gate(x_dict, edge_index_dict, h_dict)

        h_dict_new = self._calculate_hidden_state(o_dict, c_dict_new)

        return h_dict_new, c_dict_new
