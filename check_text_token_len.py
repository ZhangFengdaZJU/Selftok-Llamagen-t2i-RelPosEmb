from pathlib import Path
import numpy as np
from collections import Counter

root = Path("/home/jovyan/zfd/LlamaGen/output/t5_embs_merged")
files = sorted(root.glob("*.npy"))

shape_counter = Counter()
ok, fail = 0, 0

for f in files:
    text_token = np.load(f, mmap_mode='r')
    if text_token.shape[0] != 256:
        print(text_token.shape[0])
