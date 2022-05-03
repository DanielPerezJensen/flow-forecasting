import torch
import numpy as np
import pandas as pd
import os
import data
import torch_geometric_temporal
from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from typing import Tuple, Any, List, Optional
from pprint import pprint


def create_hetero_graph_signals(
            target_variable: str = "river_flow",
            lagged_variables: List[str] = ["river_flow"],
            freq: str = "M",
            lag: int = 6
        ) -> StaticHeteroGraphTemporalSignal:
    """
    function: create_hetero_graph_signals

    Returns a temporal dataset using torch_geometric_temporal Signal class.
    """

    edges_dict, edges_feats_dict = data.load_edges()
    feature_dicts, target_dicts = data.load_nodes(target_variable,
                                                  lagged_variables, freq, lag)

    dataset = StaticHeteroGraphTemporalSignal(edges_dict, edges_feats_dict,
                                              feature_dicts, target_dicts)

    return dataset
