#!/usr/bin/env bash

yes | conda create -n graphgps_new python=3.9
yes | conda activate graphgps_new
yes | conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
yes | conda install pyg=2.0.4 -c pyg -c conda-forge
yes | conda install openbabel fsspec rdkit -c conda-forge
yes | pip install torchmetrics
yes | pip install performer-pytorch
yes | pip install ogb
yes | pip install tensorboardX
yes | pip install wandb
yes | conda clean --all