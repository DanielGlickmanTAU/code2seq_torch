do not use sudo conda....

#!/usr/bin/env bash

#curl -- get https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh > anaconda.sh && ./anaconda.sh

#conda config --add channels https://repo.anaconda.com/pkgs/main/linux-64
#conda config --add channels https://repo.anaconda.com/pkgs/main/noarch
#conda config --add channels https://repo.anaconda.com/pkgs/r/linux-64
#conda config --add channels https://repo.anaconda.com/pkgs/r/noarch

#on aws: sudo chmod -R 777 /opt/conda


yes | conda create -n graphgps_new python=3.9
yes | conda activate graphgps_new


pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
yes | pip install torchmetrics
yes | pip install performer-pytorch
yes | pip install ogb
yes | pip install tensorboardX
yes | pip install wandb
pip install yacs
#solves some bug in tensorboardX
pip install setuptools==59.5.0
pip install commode-utils==0.4.1
pip install networkx~=2.6.3
pip install omegaconf==2.1.1

yes | conda install openbabel fsspec rdkit -c conda-forge
yes | conda clean --all