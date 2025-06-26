# build_ev_tfidf32.py
import pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from paths import *

op = pd.read_csv(RAW_DIR+"order_products__prior.csv",
                 usecols=["order_id","product_id"])
orders = pd.read_csv(RAW_DIR+"orders.csv",
                 usecols=["order_id","user_id"])

df = op.merge(orders, on="order_id")
rows, cols = df.user_id.values, df.product_id.values
data = np.ones_like(rows, np.float32)
mat = sp.coo_matrix((data, (rows, cols)),
                    shape=(df.user_id.max()+1, NUM_PROD+1))
tfidf = TfidfTransformer().fit_transform(mat).T        # product × user
svd = TruncatedSVD(n_components=32, random_state=1)
ev_tfidf32 = svd.fit_transform(tfidf)
np.save(OUT_DIR+"ev_tfidf32.npy", ev_tfidf32)
