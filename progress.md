## Reproducing  "Benchmarking Graph Neural networks" results for GNN on Pattern Dataset:

### Goal: Verify Models and Evaluation method is the same.

### Parameters Used:

number of different seeds = 4 num_layers=4; emb_dim = 110; drop_ratio=0; patience=60; residual=True;
learning_rate=[1e-3,1e-4]; ,lr_schedule_patiance=5,lr_reduce_factor=0.5

number of params: 100k

### How to Run:
train_pattern_gnn_benchmark.py
or ./main_pattern.py.py with args

### Expected Results from the paper: 85.59 std:0.011

### Results: mean:85.575 std:0.04
https://www.comet.ml/danielglickmantau/gnn-4-restore-benchmark-fixed-2/view/new/panels

