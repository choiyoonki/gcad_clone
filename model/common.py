"""
=============================================================================
models/common.py - TSMixer의 기본 빌딩 블록
=============================================================================
역할: TSMixer 모델의 핵심 구성요소인 ResBlock과 RevIN을 정의
연결: models/tsmixer.py에서 import하여 사용

논문 연결 (Section: Prediction-based Gradient Generator):
  - "The predictor consists of L stacked Mixer Predictor Layers, each
     containing interleaved temporal mixing and feature mixing MLPs."
  - "The temporal mixing MLPs are shared across all N features, while 
     the feature mixing MLPs are shared across all time steps."
  
  ResBlock = 논문의 "Mixer Predictor Layer" 하나에 해당
  RevIN = 입력 정규화/역정규화 (Reversible Instance Normalization)

구조도 (논문 Figure 1의 Mixer Predictor Layer):
  ┌─────────────────────────────────┐
  │  입력 x: (batch, seq_len, N)    │
  │         ↓                       │
  │  [BatchNorm → Temporal Linear]  │  ← 시간축 혼합 (Time Mixing)
  │         ↓ + skip connection     │
  │  [BatchNorm → Feature Linear]   │  ← 변수축 혼합 (Feature Mixing)  
  │         ↓ + skip connection     │
  │  출력: (batch, seq_len, N)      │
  └─────────────────────────────────┘
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    TSMixer의 Residual Block (= 논문의 Mixer Predictor Layer).
    
    논문 설명:
      - Temporal Mixing: 시간축(seq_len)을 따라 MLP 적용 → 시간적 패턴 학습
        "temporal mixing MLPs are shared across all N features"
      - Feature Mixing: 변수축(N)을 따라 MLP 적용 → 변수 간 관계 학습
        "feature mixing MLPs are shared across all time steps"
      - 각 단계마다 Residual Connection (잔차 연결) 사용
    
    이것이 중요한 이유:
      - 이 블록이 변수 간의 비선형 관계를 학습하는 핵심 부분
      - 나중에 이 네트워크의 그래디언트를 통해 Granger 인과관계를 추출함
      - Feature Mixing이 변수 i → 변수 j의 영향을 모델링하는 부분
    """
    
    def __init__(self, input_shape, dropout, ff_dim):
        """
        Args:
            input_shape (tuple): (seq_len, num_features) = (τ, N)
                - seq_len: 입력 시퀀스 길이 (논문의 τ)
                - num_features: 변수(센서) 수 (논문의 N)
            dropout (float): 드롭아웃 비율
            ff_dim (int): Feature Mixing의 은닉층 차원
        """
        super(ResBlock, self).__init__()
        
        # ===== Temporal Mixing (시간축 혼합) =====
        # 시간축을 따라 MLP를 적용하여 시간적 패턴을 학습
        # 입력: (batch, seq_len, N) → transpose → (batch, N, seq_len) → Linear → (batch, N, seq_len)
        self.norm1 = nn.BatchNorm1d(input_shape[0] * input_shape[1])  # 전체 flatten 후 정규화
        self.linear1 = nn.Linear(input_shape[0], input_shape[0])       # seq_len → seq_len
        self.dropout1 = nn.Dropout(dropout)
        
        # ===== Feature Mixing (변수축 혼합) =====
        # 변수축을 따라 MLP를 적용하여 변수 간 관계를 학습
        # 입력: (batch, seq_len, N) → Linear → (batch, seq_len, ff_dim) → Linear → (batch, seq_len, N)
        self.norm2 = nn.BatchNorm1d(input_shape[0] * input_shape[1])
        self.linear2 = nn.Linear(input_shape[-1], ff_dim)    # N → ff_dim (확장)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear3 = nn.Linear(ff_dim, input_shape[-1])    # ff_dim → N (복원)
        self.dropout3 = nn.Dropout(dropout)
  
    def forward(self, x):
        """
        순전파.
        
        Args:
            x: (batch, seq_len, num_features) = (batch, τ, N)
        Returns:
            출력: (batch, seq_len, num_features) = (batch, τ, N)
        """
        inputs = x  # 잔차 연결을 위해 원본 저장
        
        # ===== 1. Temporal Mixing (시간축 혼합) =====
        # BatchNorm: flatten → 정규화 → reshape
        x = self.norm1(torch.flatten(x, 1, -1)).reshape(x.shape)
        # transpose하여 시간축에 Linear 적용: (batch, τ, N) → (batch, N, τ)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.linear1(x))  # (batch, N, τ) → (batch, N, τ)
        # 다시 원래 형태로: (batch, N, τ) → (batch, τ, N)
        x = torch.transpose(x, 1, 2)
        x = self.dropout1(x)
        
        # 첫 번째 잔차 연결: Temporal Mixing 출력 + 원본 입력
        res = x + inputs

        # ===== 2. Feature Mixing (변수축 혼합) =====
        # BatchNorm
        x = self.norm2(torch.flatten(res, 1, -1)).reshape(res.shape)
        # 변수축에 2층 MLP 적용: N → ff_dim → N
        x = F.relu(self.linear2(x))   # (batch, τ, N) → (batch, τ, ff_dim)
        x = self.dropout2(x)
        
        x = self.linear3(x)           # (batch, τ, ff_dim) → (batch, τ, N)
        x = self.dropout3(x)

        # 두 번째 잔차 연결: Feature Mixing 출력 + Temporal Mixing 출력
        return x + res


# RevIN 구현 (https://github.com/ts-kim/RevIN/blob/master/RevIN.py 기반)
class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    
    역할: 시계열 데이터의 분포 변화(distribution shift)를 처리
      - 입력 시: 각 시퀀스의 평균/표준편차로 정규화 (norm)
      - 출력 시: 정규화를 역변환하여 원래 스케일로 복원 (denorm)
    
    이것이 GCAD에서 중요한 이유:
      - 센서 데이터는 각 변수마다 스케일이 다름
      - 정규화를 통해 모델이 스케일에 무관한 패턴을 학습
      - 그래디언트 기반 인과관계 추출 시 스케일 차이로 인한 편향 방지
    """
    
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Args:
            num_features (int): 변수(채널) 수 = 논문의 N
            eps (float): 수치 안정성을 위한 작은 값
            affine (bool): True면 학습 가능한 affine 파라미터 사용
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        """
        Args:
            x: 입력 텐서
            mode: 'norm' (정규화) 또는 'denorm' (역정규화)
            target_slice: denorm 시 특정 변수만 역정규화할 때 사용
        """
        if mode == 'norm':
            self._get_statistics(x)  # 평균, 표준편차 계산
            x = self._normalize(x)   # 정규화 적용
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)  # 역정규화
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """학습 가능한 affine 파라미터 초기화: weight=1, bias=0"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        입력 x의 시간축에 대한 평균과 표준편차 계산.
        
        x shape: (batch, seq_len, num_features)
        → 시간축(dim=1)에 대해 평균/표준편차 계산
        → mean, stdev shape: (batch, 1, num_features)
        """
        dim2reduce = tuple(range(1, x.ndim-1))  # 시간축 차원
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """정규화: (x - mean) / stdev * weight + bias"""
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        """
        역정규화: 정규화의 역과정.
        예측값을 원래 스케일로 복원.
        
        target_slice: MS 모드에서 특정 변수만 역정규화할 때 사용
        """
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x
