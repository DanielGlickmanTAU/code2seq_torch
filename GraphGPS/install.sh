do not use sudo conda....

#!/usr/bin/env bash

#curl -- get https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh > anaconda.sh && ./anaconda.sh

#conda config --add channels https://repo.anaconda.com/pkgs/main/linux-64
#conda config --add channels https://repo.anaconda.com/pkgs/main/noarch
#conda config --add channels https://repo.anaconda.com/pkgs/r/linux-64
#conda config --add channels https://repo.anaconda.com/pkgs/r/noarch
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

WARNING: The script shortuuid is installed in '/home/ubuntu/.local/bin' which is not on PATH.

WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(143): Could not remove or rename /opt/conda/envs/pytorch/conda-meta/numpy-1.21.3-py38he2449b9_0.json.  Please remove this file manually (you may need to reboot to free file handles)
