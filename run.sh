#!/bin/bash

echo "BASELINES"
cd baselines
echo "Index: No, Surface: No, Cloud: No"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=base training.epochs=25

echo "Index: NDSI, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-all training.epochs=25 \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False

echo "Index: NDVI, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDVI-all training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True

echo "Index: Both, Surface: Both, Cloud: Both"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-NDVI-all training.epochs=25 \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True

echo "Index: No, Surface: Both, Cloud: Both"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-NDVI-no-index training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True

echo "Index: No, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-no-index training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False

echo "Index: No, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDVI-no-index training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True

echo "Index: No, Surface: NDSI, Cloud: NDVI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-surf-NDVI-cloud training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=True

echo "Index: No, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-cloud-NDVI-surf training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=False

echo "Index: NDSI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-index-cloud-NDVI-surf training.epochs=25 \
data.ndsi.index=True data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=False

echo "Index: NDVI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W optimizer.hparams.lr=0.015 optimizer.hparams.weight_decay=0.012 model=gru model.hidden_dim=85 \
data.scaler_name=standard data.lagged_stations=[34,340,341,342] data.target_stations=[34,340,341,342] run.seeds=[34,56,67,102,105] \
run.log.wandb=False run.save=True run.name=NDSI-cloud-NDVI-index-surf training.epochs=25 \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=False


echo "GRAPH MODELS"
cd ../graph
echo "HeteroSeq"
echo "Index: No, Surface: No, Cloud: No"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=base run.save=True data.time_features=True

echo "Index: NDSI, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-all run.save=True \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False data.time_features=True

echo "Index: NDVI, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDVI-all run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: Both, Surface: Both, Cloud: Both"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-NDVI-all run.save=True \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: Both, Cloud: Both"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-NDVI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False data.time_features=True

echo "Index: No, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDVI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDSI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-surf-NDVI-cloud run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-index-NDVI-surf run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "Index: NDSI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-NDVI-surf-index run.save=True \
data.ndsi.index=True data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "Index: NDVI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-NDVI-surf run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "HeteroMLP"
echo "Index: No, Surface: No, Cloud: No"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=51 model.hidden_dim=168 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=base run.save=True data.time_features=True

echo "Index: NDSI, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-all run.save=True \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False data.time_features=True

echo "Index: NDVI, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDVI-all run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: Both, Surface: Both, Cloud: Both"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-NDVI-all run.save=True \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: Both, Cloud: Both"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-NDVI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDSI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=False data.time_features=True

echo "Index: No, Surface: NDVI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDVI-no-index run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDSI, Cloud: NDVI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-surf-NDVI-cloud run.save=True \
data.ndsi.index=False data.ndsi.surface=True data.ndsi.cloud=False \
data.ndvi.index=False data.ndvi.surface=False data.ndvi.cloud=True data.time_features=True

echo "Index: No, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-NDVI-surf run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=False data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "Index: NDSI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-index-NDVI-surf run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "Index: NDVI, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=False model=HeteroMLP \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-NDVI-surf-index run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True

echo "Homogeneous"
echo "Base"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=1 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 \
run.log.wandb=False training.gpu=1 training.batch_size=64 training.epochs=25 run.seeds=[34,56,67,102,105] \
run.name=base-homogeneous run.save=True data.time_features=True data.graph_type=homogeneous

echo "Connected"
echo "Base"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=2 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 \
run.log.wandb=False training.gpu=1 training.batch_size=64 training.epochs=25 run.seeds=[34,56,67,102,105] \
run.name=base-connected run.save=True data.time_features=True data.graph_type=connected

echo "Index: Both, Surface: Both, Cloud: Both"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-NDVI-all-connected run.save=True \
data.ndsi.index=True data.ndsi.surface=True data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=True data.time_features=True data.graph_type=connected

echo "Index: No, Surface: NDVI, Cloud: NDSI"
python train.py data.freq=W data.graph_type=base data.scaler_name=standard data.sequential=True model=HeteroSeq \
model.convolution.name=sage model.convolution.num_layers=4 model.convolution.out_channels=57 model.hidden_dim=101 \
optimizer.hparams.lr=0.00073 optimizer.hparams.weight_decay=0.00058 run.log.wandb=False training.gpu=1 training.batch_size=64 \
training.epochs=25 run.seeds=[34,56,67,102,105] run.name=NDSI-cloud-NDVI-surf-index-connected run.save=True \
data.ndsi.index=False data.ndsi.surface=False data.ndsi.cloud=True \
data.ndvi.index=True data.ndvi.surface=True data.ndvi.cloud=False data.time_features=True data.graph_type=connected