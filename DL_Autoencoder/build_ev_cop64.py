# build_ev_cop64.py
import pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from itertools import combinations
from collections import Counter
from tqdm import tqdm
from paths import *

# 1. 로드 & max id
df = pd.read_csv(RAW_DIR + "order_products__prior.csv",
                 usecols=["order_id", "product_id"])
NUM_PROD = df.product_id.max() + 1          # ← (①) 49 688 → 49 689
print("matrix dim =", NUM_PROD)

# 2. pair 카운트
cnt = Counter()
for _, grp in tqdm(df.groupby("order_id")):
    prods = sorted(set(grp.product_id))
    for i, j in combinations(prods, 2):
        cnt[(i, j)] += 1

rows, cols, data = zip(*((i, j, c) for (i, j), c in cnt.items()))

# 3. 대칭화 & COO (②)
rows = np.concatenate([rows, cols]).astype(np.int32)
cols = np.concatenate([cols, rows[:len(cols)]]).astype(np.int32)  # rows 복사본 사용
data = np.array(list(data) * 2, dtype=np.float32)

C = sp.coo_matrix((data, (rows, cols)),
                  shape=(NUM_PROD, NUM_PROD), dtype=np.float32)

# 4. PPMI
row_sum = np.array(C.sum(1)).ravel()
tot = row_sum.sum()
ppmi = C.copy().astype(np.float32)
ppmi.data = np.log((ppmi.data * tot) /
                   (row_sum[ppmi.row] * row_sum[ppmi.col]))
ppmi.data = np.maximum(ppmi.data, 0)

# 5. SVD → 64-d EV
svd = TruncatedSVD(n_components=EV_DIM, random_state=0)
ev_cop = svd.fit_transform(ppmi)

np.save(OUT_DIR + "ev_cop64.npy", ev_cop)
print("✔ saved:", OUT_DIR + "ev_cop64.npy")
