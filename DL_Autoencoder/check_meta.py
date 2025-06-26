import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1) AE 임베딩 로드
Z = np.load("ev_out/ev_final64_best.npy")[1:]  # (NUM_PROD,64)

# 2) 메타데이터 로드 & 정렬
products = pd.read_csv("raw/products.csv",
                       usecols=["product_id","aisle_id","department_id"])
products = products.sort_values("product_id").reset_index(drop=True)
aisle = products["aisle_id"].values
dept  = products["department_id"].values

# 3) train/test split & 분류
for labels, name in [(aisle,"Aisle"), (dept,"Department")]:
    Xtr, Xte, ytr, yte = train_test_split(
        Z, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print(f"\n— {name} Classification —")
    print("Accuracy:", accuracy_score(yte, ypred))
    print(classification_report(yte, ypred, zero_division=0))
