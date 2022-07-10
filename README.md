# Flow Forecasting using Graph Neural Networks
This repo contains all code relating to a currently unwritten thesis project concerning predicting river flow in the Atacama region of Chile. In this work we try to model the objective at hand as a graph to leverage spatial information about the Atacama watershed in our predictive task.

Previously as part of this work we established some simple baselines using more traditional artificial neural networks such as Multi-Layer Perceptrons, Long Short-Term Memories and Gated Recurrent Units. This work can be found in the corresponding subdirectory. Some code is shared between these two projects as some of the data loading and handling is equivalent between the two. For an interactive visualization of our work you can check out [our custom webapp](https://github.com/DanielPerezJensen/flow-forecasting-graph-webapp).

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

Installing the environment using the Pipfile contains all modules used in both the graph and baselines subdirectories.

## Running the models
To run the models simply call train.py in each of the directories. Configuration options are detailed in the respective config folders. To for example run a graph model using the standard scaler, while including ndsi index and surface data simply do:

```bash
cd graph
python train.py data.scaler_name=standard data.ndsi.index=True data.ndsi.surface=True
```

To save the run, indicate it using `run.save=True` and name it using `run.name=<some-name>`, the name by default is base. To run across multiple seeds, specifiy a list of seeds `run.seeds=[1,2,3,4,5]`

Experiments are saved in the experiments folder, we save the outputs and targets for the validation and test set, and a metrics.txt which indicates the RMSE and NSE for each seed. 

To train all the models used in the results section please run `./run.sh`. This will save all experiments showed in the tables in experiments/baselines and experiments/graphs. To run the non-parametric heuristics also displayed in the results, please run 
```bash
cd baselines
python heuristics.py --model AverageMonth
python heuristics.py --model PreviousMonth
```

To generate the results you can use create_figure.py. To create all the plots displayed in the results, you should run:
```bash
python create_figure.py --files baselines/gru-base graph/HeteroMLP-base graph/HeteroSeq-base heuristics/AverageMonth heuristics/PreviousMonth --labels bGRU gHMLP gHSeq bAvg bPrev --plot_all --plot_scatter --plot_stations --save_dir all_models
```

Any other experiment can also be plotted in this way by specifying the run name and its label.
