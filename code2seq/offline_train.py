import os
import sys

os.system("nohup sh -c '" +
          # sys.executable + " compression_flows.py> res.txt " +
          # sys.executable + " code2seq_wrapper.py -c ../config/code2seq-py150k.yaml train> training150.txt " +
          sys.executable + " code2seq_wrapper.py -c ../config/code2seq-py150k-compressed-50.yaml train> training50.txt " +
          "' &")
