# build_ev_meta16.py
import pandas as pd, numpy as np
from paths import *

prod = pd.read_csv(RAW_DIR+"products.csv",
                   usecols=["product_id","aisle_id","department_id"])
num_dept  = prod.department_id.max()+1
num_aisle = prod.aisle_id.max()+1

dept_emb  = np.random.normal(0,0.1,(num_dept,8)).astype(np.float32)
aisle_emb = np.random.normal(0,0.1,(num_aisle,8)).astype(np.float32)

ev_meta = np.empty((NUM_PROD+1,16),np.float32)
for _,r in prod.iterrows():
    ev_meta[r.product_id,:8]  = dept_emb[r.department_id]
    ev_meta[r.product_id,8:]  = aisle_emb[r.aisle_id]
np.save(OUT_DIR+"ev_meta16.npy", ev_meta)
