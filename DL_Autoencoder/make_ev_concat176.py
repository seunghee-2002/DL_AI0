import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from paths import *

# 1) 원본 임베딩 로드
ev_cop   = np.load(OUT_DIR+"ev_cop64.npy")   # (N+1,64)
ev_seq   = np.load(OUT_DIR+"ev_seq64.npy")   # (N+1,64)
ev_meta  = np.load(OUT_DIR+"ev_meta16.npy")  # (N+1,16)
try:
    ev_tfidf = np.load(OUT_DIR+"ev_tfidf32.npy")  # (N+1,32)
except FileNotFoundError:
    ev_tfidf = np.zeros((NUM_PROD+1,32), np.float32)

# 2) 블록별 L2 정규화 (각 행을 unit‐vector로)
ev_cop_norm   = normalize(ev_cop,   axis=1)  # shape (N+1,64)
ev_seq_norm   = normalize(ev_seq,   axis=1)  # shape (N+1,64)
ev_tfidf_norm = normalize(ev_tfidf, axis=1)  # shape (N+1,32)
ev_meta_norm  = normalize(ev_meta,  axis=1)  # shape (N+1,16)

###
#[Epoch 095] Train MSE: 0.0020 | Val MSE: 0.0020, MAE: 0.0333, R²: 0.7398, EV: 0.7408
#[Epoch 097] Train MSE: 0.0020 | Val MSE: 0.0021, MAE: 0.0337, R²: 0.7388, EV: 0.7411
#[Epoch 098] Train MSE: 0.0020 | Val MSE: 0.0020, MAE: 0.0332, R²: 0.7401, EV: 0.7413
#[Epoch 099] Train MSE: 0.0020 | Val MSE: 0.0020, MAE: 0.0332, R²: 0.7404, EV: 0.7415
#[Epoch 100] Train MSE: 0.0020 | Val MSE: 0.0020, MAE: 0.0331, R²: 0.7408, EV: 0.7417
###

# 3) (선택) 또는 블록별 표준화      
scaler = StandardScaler()
ev_cop_std   = scaler.fit_transform(ev_cop)
ev_seq_std   = scaler.fit_transform(ev_seq)
ev_tfidf_std = scaler.fit_transform(ev_tfidf)
ev_meta_std  = scaler.fit_transform(ev_meta)

###
#[Epoch 096] Train MSE: 0.1840 | Val MSE: 0.3134, MAE: 0.2978, R²: 0.6928, EV: 0.6939
#[Epoch 097] Train MSE: 0.1831 | Val MSE: 0.3128, MAE: 0.2974, R²: 0.6934, EV: 0.6944
#[Epoch 098] Train MSE: 0.1821 | Val MSE: 0.3125, MAE: 0.2972, R²: 0.6937, EV: 0.6948
#[Epoch 099] Train MSE: 0.1813 | Val MSE: 0.3128, MAE: 0.2973, R²: 0.6936, EV: 0.6953
#[Epoch 100] Train MSE: 0.1811 | Val MSE: 0.3122, MAE: 0.2968, R²: 0.6942, EV: 0.6959


# 4) 정규화된 블록들을 이어 붙이기
X = np.hstack([
    ev_cop_norm,
    ev_seq_norm,
    ev_tfidf_norm,
    ev_meta_norm
])
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)
# 5) 저장
np.save(OUT_DIR+"ev_concat176.npy", X_std)
print("✔ saved:", OUT_DIR+"ev_concat176.npy")
