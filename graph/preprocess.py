import pandas as pd
import numpy as np
import os
from os.path import join
import json
from typing import Tuple

from scipy.spatial import distance


def edge_features(source: Tuple[float, float],
                  target: Tuple[float, float]) -> Tuple[int, float, float]:
    """
    function: edge_features

    Calculates and returns the features edges between
    source and target nodes, namely the distance, x and y difference.
    """

    x_diff = source[0] - target[0]
    y_diff = source[1] - target[1]

    dist = round(distance.euclidean(source, target))

    return dist, x_diff, y_diff


def preprocess_static_data() -> None:
    """
    function: preprocess_static_data

    Preprocesses all static information relating to the nodes and edges
    of our graph. Namely the subsubwatershed nodes, the measurement nodes,
    the edges between subsubwatersheds, the edges between measurement nodes,
    and the edges between subsubwatersheds and measurement nodes. These are
    all stored to disk.
    """
    subsub_static_data = preprocess_subsub_static_data()
    static_subsub_nodes_df, static_subsub_edges_df = subsub_static_data

    measurement_static_data = preprocess_measurement_static_data()
    static_msr_nodes_df, static_msrmsr_edges_df = measurement_static_data

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
                enumerate(static_msr_nodes_df.index.to_list())}

    static_subsub_nodes_df.index = static_subsub_nodes_df.index.map(s_mapper)
    static_msr_nodes_df.index = static_msr_nodes_df.index.map(m_mapper)

    static_subsub_edges_df.src = static_subsub_edges_df.src.map(s_mapper)
    static_subsub_edges_df.dst = static_subsub_edges_df.dst.map(s_mapper)

    static_msrmsr_edges_df.src = static_msrmsr_edges_df.src.map(m_mapper)
    static_msrmsr_edges_df.dst = static_msrmsr_edges_df.dst.map(m_mapper)

    static_submsr_edges_df.src = static_submsr_edges_df.src.map(s_mapper)
    static_submsr_edges_df.dst = static_submsr_edges_df.dst.map(m_mapper)

    # Below we save our data into processed folder path
    processed_path = join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(join(processed_path, "static"), exist_ok=True)
    os.makedirs(join(processed_path, "graph"), exist_ok=True)
    os.makedirs(join(processed_path, "graph", "base"), exist_ok=True)
    os.makedirs(join(processed_path, "temporal"), exist_ok=True)

    static_subsub_nodes_df.to_csv(join(processed_path, "static",
                                       "subsub.csv"))
    static_msr_nodes_df.to_csv(join(processed_path, "static",
                                    "measurement.csv"))

    static_subsub_edges_df.to_csv(join(processed_path, "graph", "base",
                                       "subsub-flows-subsub.csv"))
    static_msrmsr_edges_df.to_csv(join(processed_path, "graph", "base",
                                       "measurement-flows-measurement.csv"))
    static_submsr_edges_df.to_csv(join(processed_path, "graph", "base",
                                       "subsub-in-measurement.csv"))

    # We store the mappers as well
    subsub_path = join(processed_path, "static", "subsub.json")
    measurement_path = join(processed_path, "static", "measurement.json")
    with open(subsub_path, "w") as s_f, open(measurement_path, "w") as m_f:
        json.dump(s_mapper, s_f)
        json.dump(m_mapper, m_f)


def preprocess_subsub_static_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    function: preprocess_subsub_static_data

    Preprocesses the static data about subsubwatershed nodes
    and edges between subsubwatershed nodes into properly formatted .csv files.
    """
    unprocessed_path = join("data", "unprocessed")
    static_subsub_path = join(unprocessed_path, "Data_Static",
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

        source_xy = (node_feature_dict[source]["Centroid_X"],
                     node_feature_dict[source]["Centroid_Y"])
        target_xy = (node_feature_dict[target]["Centroid_X"],
                     node_feature_dict[target]["Centroid_Y"])

        dist, x_diff, y_diff = edge_features(source_xy, target_xy)
        edge_feature_list.append([source, target, dist])

    static_subsub_edges_df = pd.DataFrame(edge_feature_list,
                                          columns=["src", "dst", "distance"])

    return static_subsub_nodes_df, static_subsub_edges_df


def preprocess_measurement_static_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    function: preprocess_measurement_static_data

    Preprocesses and returns the static data about measurement
    nodes and edges between measurement nodes into formatted .csv files.
    """
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

        source_xy = (node_feature_dict[source]["Centroid_X"],
                     node_feature_dict[source]["Centroid_Y"])
        target_xy = (node_feature_dict[target]["Centroid_X"],
                     node_feature_dict[target]["Centroid_Y"])

        dist, x_diff, y_diff = edge_features(source_xy, target_xy)
        edge_feature_list.append([source, target, dist])

    static_msrmsr_edges_df = pd.DataFrame(edge_feature_list,
                                          columns=["src", "dst", "distance"])

    return static_measurement_nodes_df, static_msrmsr_edges_df


def preprocess_temporal_data() -> None:
    """
    function: preprocess_temporal_data

    Preprocesses the temporal river flow and ndsi/ndvi data into
    properly formatted csv files and saves it to disk.
    """
    unprocessed_path = join("data", "unprocessed")
    processed_path = join("data", "processed")

    flow_data_folder = "Data_RiverFlow"
    flow_data_file = "Caudales.txt"

    date_columns = ["day", "month", "year", "hour"]

    df = pd.read_csv(join(unprocessed_path, flow_data_folder, flow_data_file),
                     delimiter="\t", index_col=False,
                     names=["station_number", "day", "month", "year", "hour",
                            "river_height", "river_flow",
                            "information", "origin"])

    # Convert date go datetime and add as column
    date = pd.to_datetime(dict(year=df.year, month=df.month,
                               day=df.day, hour=df.hour))
    df = df.drop(columns=date_columns)
    df.insert(1, 'date', date)

    # We gather our mapper for the nodes so index is preserved
    with open(join(processed_path, "static", "measurement.json"), "r") as f:
        nodes_mapper = json.load(f)

    nodes_mapper = {int(k): v for k, v in nodes_mapper.items()}

    df.station_number = df.station_number.map(nodes_mapper)

    # Save dataframe to disk
    df.to_csv(join(processed_path, "temporal", "measurements.csv"))

    # Take care of the NDSI NDVI data
    data_folder = "Data_NDSI_NDVI"
    data_files = ["NDSI.txt", "NDVI.txt"]

    # We gather our mapper for the nodes so index is preserved
    with open(join(processed_path, "static", "subsub.json"), "r") as f:
        subsub_mapper = json.load(f)

    for data_file in data_files:
        df = pd.read_csv(join(unprocessed_path, data_folder, data_file),
                         delimiter="\t", index_col=False,
                         names=["Watershed", "Subsubwatershed", "Product",
                                "Date", "Areaini", "Areareproj", f"Surfmax",
                                f"Surfmin", "Surfavg", "max", "min", f"avg",
                                "Surfcloudmax", "Surfcloudmin", "Surfcloudavg",
                                "Surfbadpixmax", "Surfbadpixmin",
                                "Surfbadpixavg"],
                         dtype={"Subsubwatershed": str})

        # We only care about copiapo watershed
        df = df.loc[df.Watershed == "Atacama_Copiapo"]

        # We only care about the subsubwatersheds found in our mapper
        df = df.loc[df.Subsubwatershed.isin(subsub_mapper)]

        # Convert date to datetime and convert to wateryears
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop("Date", axis=1)

        # Map subsubwatersheds to the correct index
        df.Subsubwatershed = df.Subsubwatershed.map(subsub_mapper)

        df.to_csv(join(processed_path, "temporal", f"{data_file[:4]}.csv"))


if __name__ == "__main__":
    preprocess_static_data()
    preprocess_temporal_data()
