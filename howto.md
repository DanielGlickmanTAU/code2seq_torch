## profile:
run with --profiler True

export trace with "prof.export_chrome_trace("trace.json")" in custom_train#_profile and upload with res = run.save('
trace.json, policy='now')  
view it in chrome://tracing

when want to decorate specific functions, do with torch.autograd.profiler.record_function("label-z")



## load model from checkpoint
1) download ckpt file e.g https://wandb.ai/daniel-ai/single-shape-coloring-rows-shapes-visualization/runs/dmtrvo2u/files/tests/results/1657732829.8918536_0/ckpt
2) put it under dir named ckpt. e.g runs/ckpt/1499.ckpt  
3) run main with flag --load_checkpoint_from_dir runs (where runs is the dir containins ckpt), it will load .ckpt file with highest number
notice to use the right parameters to load model, can take that from wandb overview


## visualize attention
1) break point at a place where you have attentio weights e.g gps layer after using attention module
2) visualization.draw_attention(batch[index_in_batch].graph, node_id, att_weights[index_in_batch])