import os
import sys
import random

max_word_joins = "1"
vocab_size = "100"
file_name = f"res{vocab_size}_{max_word_joins}.txt"
sleep = 60 * 60 * 8 * 6
print(file_name)
os.system("nohup sh -c " + "'" +
          sys.executable + " compress_script.py --max_word_joins " + max_word_joins + " --vocab_size " + vocab_size + " > " + file_name + " " +
          # sys.executable + " code2seq_wrapper.py -c ../config/code2seq-py150k.yaml train> training.txt " +
          "' &")
