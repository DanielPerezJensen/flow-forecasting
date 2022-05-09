# Flow Forecasting using Graph Neural Networks
This repo contains all code relating to a currently unwritten thesis project concerning predicting river flow in the Atacama region of Chile. In this work we try to model the objective at hand as a graph to leverage spatial information about the Atacama watershed in our predictive task. We implement graph neural networks using functionality from [PyTorch](https://github.com/pytorch/pytorch), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal). 

Previously as part of this work we established some simple baselines using more traditional artificial neural networks such as Multi-Layer Perceptrons, Long Short-Term Memories and Gated Recurrent Units. This work can be found in this [Flow Forecasting Baselines](https://github.com/DanielPerezJensen/flow-forecasting-baselines). Some code is shared between these two projects as some of the data loading and handling is equivalent between the two. For an interactive visualization of our work you can check out [our custom webapp](https://github.com/DanielPerezJensen/flow-forecasting-graph-webapp).

## Installation Instructions
The development environment of this work can be generated using [PipEnv](https://pipenv.pypa.io/en/latest/). Simply run `pipenv install` when in the working directory of the repository. Do keep in mind that development was done on a Linux machine and some of the packages listed in the Pipfile assume a Linux machine. To work with this on other operating systems please update the following package lines:
```
torch = {index="downloadpytorch", version="==1.11.0+cu113"}
...
...
torch-scatter = {file = "https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl"}
torch-sparse = {file = "https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl"}
torch-cluster = {file = "https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl"}
torch-spline-conv = {file = "https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl"}
```
You can check the package index and find the correct version for your machine, and replace the link in the file parameter.

## Directory Structure 
```
├── data
│   ├── processed    
|   |   ├── graph
|   |   │   ├── base
|   |   |   |    ├── measurement-flows-measurement.csv  # Edge information for edges between measurement and measurement nodes
|   |   |   |    ├── subsub-flows-subsub.csv            # Edge information for edges between subsubwatershed and subsubwatershed nodes
|   |   |   |    └── subsub-in-measurement.csv          # Edge information for edges between subsubwatershed and measurement nodes
|   |   |   ├── connected
|   |   |   |    └── Same structure as `base` directory
|   |   |   ├── homogeneous
|   |   |   |    └── Same structure as `base` directory
|   │   ├── static
|   |   |   ├── measurement.csv                         # Static node information for measurement nodes
|   |   |   ├── subsub.csv                              # Static node information for subsubwatershed nodes
|   |   |   ├── measurement.json                        # Mapping dictionary for measurement nodes
|   |   |   └── subsub.json                             # Mapping dictionary for subsubwatershed nodes
|   |   └── temporal
|   |        └── raw-measurements.csv                   # Raw river flow measurements through time
|   ├── unprocessed                                     # Unprocessed data, handled by preprocess.py
|   |   ├── Data_NDSI_NDVI                              # NDSI and NDVI data
|   |   |   ├── NDSI.txt
|   |   |   └── NDVI.txt
|   |   ├── Data_RiverFlow                              # River Flow data
|   |   |   └── Caudales.txt
|   └───└── Data_Static                                 # Static information about nodes in our graph
|           └── DataCriosphere-Watershedf.txt
├── base_models                                         # All base models written using PyTorch and PyTorch Geometric
|   ├── __init__.py                                     # Module initialization code
|   └── HeteroGLSTM.py                                  # HeteroGLSTM module definition
├── config                                              # Configuration files
|   ├── optimizer.py
|   |   └── adam.yaml
|   └── config.yaml                                     # HeteroGLSTM module definition
|
├── data.py                                             # Data handling code
├── models.py                                           # PyTorch Lightning models
├── preprocess.py                                       # Preprocessing code
├── train.py                                            # Training code
├── Pipfile                                             # Environment configuration
├── Pipfile.lock                                        # Environment configuration
├── .gitignore   
└── README.md                                           # This file :)
```

## Other stuff
TODO
