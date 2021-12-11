import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " compression_flows.py --vocab_size 100> res100.txt " +
          # sys.executable + " code2seq_wrapper.py -c ../config/code2seq-py150k.yaml train> training.txt " +
          "' &")
