import pandas as pd
import numpy as np
import os
import json
from typing import Tuple

from scipy.spatial import distance


def edge_features(source: Tuple[float, float],
                  target: Tuple[float, float]) -> Tuple[int, float, float]:

    x_diff = source[0] - target[0]
    y_diff = source[1] - target[1]

    dist = round(distance.euclidean(source, target))

    return dist, x_diff, y_diff


def preprocess_static_data() -> None:
    subsub_static_data = preprocess_subsub_static_data()
    static_subsub_nodes_df, static_subsub_edges_df = subsub_static_data

    measurement_static_data = preprocess_measurement_static_data()
    static_measurement_nodes_df, static_msrmsr_edges_df = subsub_static_data

    # Here we define edges between subsub nodes and measurement station nodes
    # Handled separately to enable different features to be added down the line
    edge_list = [("03404", "340"), ("03414", "341"), ("03421", "342")]

    edge_feature_list = []

    for edge in edge_list:
        source, target = edge
        edge_feature_list.append([source, target])

    static_submsr_edges_df = pd.DataFrame(edge_feature_list,
                                          columns=["src", "dst"])

    # We map the various nodes to an index starting at 0
    s_mapper = {node: i for i, node in
                enumerate(static_subsub_nodes_df.index.to_list())}
    m_mapper = {node: i for i, node in
                enumerate(static_measurement_nodes_df.index.to_list())}

    static_subsub_nodes_df.index = static_subsub_nodes_df.index.map(s_mapper)
    static_measurement_nodes_df.index = static_measurement_nodes_df.index.map(m_mapper)

    static_subsub_edges_df.src = static_subsub_edges_df.src.map(s_mapper)
    static_subsub_edges_df.dst = static_subsub_edges_df.dst.map(s_mapper)

    static_msrmsr_edges_df.src = static_msrmsr_edges_df.src.map(m_mapper)
    static_msrmsr_edges_df.dst = static_msrmsr_edges_df.dst.map(m_mapper)

    static_submsr_edges_df.src = static_submsr_edges_df.src.map(s_mapper)
    static_submsr_edges_df.dst = static_submsr_edges_df.dst.map(m_mapper)

    # Below we save our data into processed folder path
    processed_path = os.path.join("data", "processed")

    static_subsub_nodes_df.to_csv(os.path.join(processed_path, "subsub.csv"))
    static_measurement_nodes_df.to_csv(os.path.join(processed_path, "measurement.csv"))

    static_subsub_edges_df.to_csv(os.path.join(processed_path, "subsub-flows-subsub.csv"))
    static_msrmsr_edges_df.to_csv(os.path.join(processed_path, "measurement-flows-measurement.csv"))
    static_submsr_edges_df.to_csv(os.path.join(processed_path, "subsub-in-measurement.csv"))

    # We store the mappers as well
    subsub_path = os.path.join(processed_path, "subsub.json")
    measurement_path = os.path.join(processed_path, "measurement.json")
    with open(subsub_path, "w") as s_f, open(measurement_path, "w") as m_f:
        json.dump(s_mapper, s_f)
        json.dump(m_mapper, m_f)


def preprocess_subsub_static_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load our unprocessed subsub node information
    unprocessed_path = os.path.join("data", "unprocessed")
    static_subsub_path = os.path.join(unprocessed_path, "Data_Static",
                                      "DataCriosphere-Watershedf.txt")

    static_subsub_nodes_df = pd.read_csv(
            static_subsub_path,
            delimiter="\t", index_col="node", na_values=-99999.000000,
            names=["node", "AreaWatershed", "PerimeterWatershed",
                   "GeometricalFactor", "Centroid_X", "Centroid_Y",
                   "MeanAltitude", "StdAltitude",
                   "MeanSlope", "StdSlope", "MeanDirSlope", "StdDirSlope",
                   "0Area", "0MeanAltitude", "0MeanDir", "0MeanSlope",
                   "1Area", "1MeanAltitude", "1MeanDir", "1MeanSlope",
                   "2Area", "2MeanAltitude", "2MeanDir", "2MeanSlope",
                   "3Area", "3MeanAltitude", "3MeanDir", "3MeanSlope",
                   "4Area", "4MeanAltitude", "4MeanDir", "4MeanSlope"],
            dtype={"Centroid_X": float, "Centroid_Y": float}
        )

    static_subsub_nodes_df.index = (static_subsub_nodes_df.index.astype(float)
                                    .astype(int).astype(str))
    static_subsub_nodes_df.index = static_subsub_nodes_df.index.str.zfill(5)
    static_subsub_nodes_df = static_subsub_nodes_df.fillna(-1)

    # Load our edges between subsub nodes
    edge_list = [("03420", "03421"), ("03413", "03414"), ("03410", "03414"),
                 ("03412", "03414"), ("03411", "03414"), ("03400", "03401"),
                 ("03401", "03404"), ("03402", "03403"), ("03403", "03404")]

    node_feature_dict = static_subsub_nodes_df.to_dict("index")

    edge_feature_list = []

    for edge in edge_list:
        source, target = edge

        source = (node_feature_dict[source]["Centroid_X"],
                  node_feature_dict[source]["Centroid_Y"])
        target = (node_feature_dict[target]["Centroid_X"],
                  node_feature_dict[target]["Centroid_Y"])

        dist, x_diff, y_diff = edge_features(source, target)
        edge_feature_list.append([source, target, dist])

    static_subsub_edges_df = pd.DataFrame(edge_feature_list,
                                          columns=["src", "dst", "distance"])

    return static_subsub_nodes_df, static_subsub_edges_df


def preprocess_measurement_static_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # We hard code the measurement node information
    measurement_nodes = [["34", 6902002, 403928, 1155],
                         ["340", 6897284, 405766, 1240],
                         ["341", 6892589, 407374, 1320],
                         ["342", 6884782, 401222, 1520]]

    static_measurement_nodes_df = pd.DataFrame(
        measurement_nodes,
        columns=["node", "Centroid_X",
                 "Centroid_Y", "altitude"]
            ).set_index("node")

    # Here we define edges between measurement station nodes
    edge_list = [("340", "34"), ("341", "34"), ("342", "34")]

    node_feature_dict = static_measurement_nodes_df.to_dict("index")

    edge_feature_list = []

    for edge in edge_list:
        source, target = edge

        source = (node_feature_dict[source]["Centroid_X"],
                  node_feature_dict[source]["Centroid_Y"])
        target = (node_feature_dict[target]["Centroid_X"],
                  node_feature_dict[target]["Centroid_Y"])

        dist, x_diff, y_diff = edge_features(source, target)
        edge_feature_list.append([source, target, dist])

    static_msrmsr_edges_df = pd.DataFrame(edge_feature_list,
                                          columns=["src", "dst", "distance"])

    return static_measurement_nodes_df, static_msrmsr_edges_df


if __name__ == "__main__":
    preprocess_static_data()
