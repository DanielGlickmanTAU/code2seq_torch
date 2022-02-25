## Reproducing  "Benchmarking Graph Neural networks" results for GNN on Pattern Dataset:

### Goal: Verify Models and Evaluation method is the same.

### Parameters Used:

number of different seeds = 4 num_layers=4; emb_dim = 80; drop_ratio=0; patience=120; residual=4;
learning_rate=[1e-3,1e-4]
,lr_schedule_patiance=5,lr_reduce_factor=0.5

### How to Run:

'/ogb-master/examples/graphproppred/mol/main_pattern.py --attention_type position
--num_transformer_layers 1 --num_layer 1 --adj_stacks 0 1 2 3 4 5

### Results: mean:81.973 std:0.580

### Expected Results from the paper: 85.59 std:0.011

### Complete? Not really

### Directions: learning rate not halfing??