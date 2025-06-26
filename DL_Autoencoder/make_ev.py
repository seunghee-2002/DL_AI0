import torch
import torch.nn as nn
import numpy as np
from paths import *  # OUT_DIR이 정의된 모듈

# 1) AE 모델 클래스 정의 (train 때와 동일하게)
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(176, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            
            nn.Linear(64, 32),   # bottleneck
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 176),
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

# 2) 모델 생성 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)

# 3) 체크포인트 로드
checkpoint_path = OUT_DIR + "ae_final32_best.pt"
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model.eval()

# 4) (선택) ev_concat176_norm.npy 로드해서 embedding 뽑기
X = np.load(OUT_DIR + "ev_concat176.npy").astype(np.float32)  # (NUM_PROD+1,176)
X_input = torch.tensor(X, device=device)
with torch.no_grad():
    _, Z = model(X_input)   # Z: (NUM_PROD+1,64)
Z = Z.cpu().numpy()

# 5) 저장
np.save(OUT_DIR + "ev_final32_best.npy", Z)
print("✔ Loaded best AE and saved embeddings to ev_final64_best.npy")
