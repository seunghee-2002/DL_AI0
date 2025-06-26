import numpy as np
from paths import *
FILE = 'ev_64' + '.npy'
ev = np.load(OUT_DIR + FILE, mmap_mode="r")
print(ev.shape)
print("file:" + FILE)
print(ev[:5,])
