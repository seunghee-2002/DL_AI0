# build_ev_seq64.py
import pandas as pd, gensim, numpy as np
from paths import *

op = pd.read_csv(RAW_DIR+"order_products__prior.csv",
                 usecols=["order_id","product_id","add_to_cart_order"])
sentences = (op.sort_values(["order_id","add_to_cart_order"])
               .groupby("order_id")["product_id"].apply(list))

w2v = gensim.models.Word2Vec(
        sentences, vector_size=EV_DIM, window=5, min_count=2,
        sg=1, workers=8, epochs=10)
ev_seq = np.zeros((NUM_PROD+1, EV_DIM), np.float32)
for pid in w2v.wv.index_to_key:
    ev_seq[int(pid)] = w2v.wv[pid]
np.save(OUT_DIR+"ev_seq64.npy", ev_seq)
