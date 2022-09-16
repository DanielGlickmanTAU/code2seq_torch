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

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+$cu113.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+$cu113.html
pip3 install torch-geometric
yes | pip install torchmetrics
yes | pip install performer-pytorch
yes | pip install ogb
yes | pip install tensorboardX
yes | pip install wandb

#solves some bug in tensorboardX
pip install setuptools==59.5.0

yes | conda install openbabel fsspec rdkit -c conda-forge
yes | conda clean --all



