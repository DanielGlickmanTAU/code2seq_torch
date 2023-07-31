# TITLE

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2205.12454-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2205.12454&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recipe-for-a-general-powerful-scalable-graph/graph-regression-on-zinc&#41;]&#40;https://paperswithcode.com/sota/graph-regression-on-zinc?p=recipe-for-a-general-powerful-scalable-graph&#41;)


This repo is based on [GraphGPS](https://arxiv.org/abs/2205.12454), which is built using [PyG](https://www.pyg.org/)
and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).

### Conda environment setup

```bash
conda create -n graphdiffuser python=3.9
conda activate graphdiffuser
conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge
conda install openbabel fsspec rdkit -c conda-forge
pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb
conda clean --all
```

### Running GraphGPS

```bash

# Running Graph Diffuser on ZINC.
python main.py --cfg configs/GPS/zinc-diffuser.yaml  wandb.use False
#molpcba
python main.py --cfg configs/GPS/ogbg-molpcba-diffuser.yaml  wandb.use False
#ppa
python main.py --cfg configs/GPS/ogbg-ppa-diffuser.yaml  wandb.use False

```
