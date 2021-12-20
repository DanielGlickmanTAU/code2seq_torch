import os
import sys
import random

file_name = f"res{random.randint(1, 10000)}.txt"
print(file_name)
os.system("nohup sh -c '" +
          sys.executable + " compress_script.py --max_node_joins 1 --vocab_size 10 > " + file_name + " " +
          # sys.executable + " code2seq_wrapper.py -c ../config/code2seq-py150k.yaml train> training.txt " +
          "' &")
