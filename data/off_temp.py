import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " py150k_extractor.py > extractor50.txt" +
          "' &")
