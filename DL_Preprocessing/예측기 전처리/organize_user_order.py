import pandas as pd
import numpy as np
from collections import defaultdict
import json

# ---------- 1) CSV 로드(메모리 절약형 dtypes) ----------
ADDR_ORDERS          = r"archive\orders.csv"
ADDR_PROD_PRIOR      = r"archive\order_products__prior.csv"
ADDR_PROD_TRAIN      = r"archive\order_products__train.csv"

dtype_orders = {
    "eval_set"              : str,
    "order_id"              : np.int32,
    "user_id"               : np.int32,
    "order_number"          : np.int16,
    "order_dow"             : np.int8,
    "order_hour_of_day"     : np.int8,
    "days_since_prior_order": np.float32,  # NaN 포함
}

dtype_order_products = {
    "order_id"          : np.int32,
    "product_id"        : np.int32,
    "add_to_cart_order" : np.int16,
    "reordered"         : np.int8,
}

df_orders            = pd.read_csv(ADDR_ORDERS,       dtype=dtype_orders)
df_prod_prior        = pd.read_csv(ADDR_PROD_PRIOR,   dtype=dtype_order_products)
df_prod_train        = pd.read_csv(ADDR_PROD_TRAIN,   dtype=dtype_order_products)
data=df_prod_train["order_id"]==1455788
print(df_prod_train[data])

# ---------- 2) prior + train 합치고 정렬 ----------
df_prods = pd.concat([df_prod_prior, df_prod_train], ignore_index=True)
df_prods.sort_values(["order_id", "add_to_cart_order"], inplace=True)  # 카트 담은 순서 유지

# ---------- 3) order_id → {products: [...], is_reordered: [...]} ----------
order_items = (
    df_prods
    .groupby("order_id")
    .agg(products     = ("product_id", list),
         is_reordered = ("reordered",  list))
    .to_dict(orient="index")              # C-레벨에서 리스트까지 만들어 줌 → 매우 빠름
)

# ---------- 4) user_id → order_number → {...} ----------
orders_by_user = {}
for user_id, g in df_orders.groupby("user_id"):
    user_dict = {
        row.order_number: {
            "eval_set"              : row.eval_set,
            "order_id"              : row.order_id,
            "order_dow"             : row.order_dow,
            "order_hour_of_day"     : row.order_hour_of_day,
            "days_since_prior_order": row.days_since_prior_order,
            **order_items.get(row.order_id, {"products": [], "is_reordered": []})
        }
        for row in g.itertuples(index=False)
    }
    orders_by_user[user_id] = user_dict

# ---------- 5) JSON 저장 ----------
with open("organized_user_order.json", "w", encoding="utf-8") as f:
    json.dump(orders_by_user, f)          # 키가 모두 문자열로 직렬화됨

# ---------- 6) 검증 ----------
with open("organized_user_order.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("총 사용자 수:", len(data))
print("사용자 500번 예시:", data.get("500") or data.get(500))  # 문자열·정수 둘 다 확인
