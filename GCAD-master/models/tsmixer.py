"""
=============================================================================
models/tsmixer.py - TSMixer 예측 모델 (= 논문의 Gradient Generator)
=============================================================================
역할: GCAD의 핵심 예측기 (Prediction-based Gradient Generator)
연결: main.py에서 모델 생성, test.py에서 그래디언트 추출에 사용

논문 연결 (Section: Prediction-based Gradient Generator):
  - "The widely studied Mixer predictor (Chen et al. 2023) is used as 
     the gradient generator in our GCAD framework."
  - "The predictor consists of L stacked Mixer Predictor Layers"
  - "The output of each layer is fed through skip connections into a 
     fully connected layer to produce the predictive output."

이 모델의 이중 역할:
  1. 학습 단계: 일반적인 시계열 예측기로 동작 (MSE 손실로 학습)
     → 정상 데이터의 변수 간 관계를 학습
  2. 테스트 단계: 그래디언트 생성기로 동작
     → 학습된 네트워크의 그래디언트를 통해 Granger 인과관계 추출

모델 구조 (논문 Figure 1):
  입력 X_{t-1}: (batch, τ, N)
       ↓
  [RevIN 정규화]
       ↓
  [Mixer Predictor Layer 1]  ← ResBlock
  [Mixer Predictor Layer 2]  ← ResBlock
       ...
  [Mixer Predictor Layer L]  ← ResBlock (L = n_block)
       ↓
  [Transpose → FC Layer → Transpose]  ← 시간축 τ → pred_len 변환
       ↓
  [RevIN 역정규화]
       ↓
  출력 ŷ_t: (batch, pred_len, N)

참고 논문: TSMixer (Chen et al., 2023) - "TSMixer: An All-MLP Architecture 
           for Time Series Forecasting"
=============================================================================
"""

import torch
import torch.nn as nn

from models.common import RevIN
from models.common import ResBlock


class TSMixerRevIN(nn.Module):
    """
    RevIN이 적용된 TSMixer 모델.
    
    논문에서의 역할:
      - 학습 시: 예측 함수 f를 학습 → ŷ_t = f(X_{t-1})
      - 테스트 시: 채널별 그래디언트 ∂L_{t,j}/∂x_{t',i}를 계산하는 기반 네트워크
      
    핵심 포인트:
      - 이 네트워크가 학습한 f의 내부 구조(그래디언트)가 
        변수 간 Granger 인과관계를 반영함
      - "The gradients of a network reflect its internal structure to some extent"
    """

    def __init__(self, input_shape, pred_len, n_block, dropout, ff_dim, target_slice):
        """
        Args:
            input_shape (tuple): (seq_len, num_features) = (τ, N)
                - seq_len (τ): Granger 인과관계의 최대 시간 지연
                - num_features (N): 센서/변수 수
            pred_len (int): 예측 길이 (보통 1)
            n_block (int): Mixer Predictor Layer 수 = 논문의 L
            dropout (float): 드롭아웃 비율
            ff_dim (int): Feature Mixing의 은닉층 차원
            target_slice: 예측 대상 변수 슬라이스 (M 모드에서는 전체)
        """
        super(TSMixerRevIN, self).__init__()
        
        self.target_slice = target_slice
        
        # RevIN: 입력 정규화 / 출력 역정규화
        self.rev_norm = RevIN(input_shape[-1])
        
        # L개의 Mixer Predictor Layer (ResBlock) 스택
        # 논문: "The predictor consists of L stacked Mixer Predictor Layers"
        self.res_blocks = nn.ModuleList(
            [ResBlock(input_shape, dropout, ff_dim) for _ in range(n_block)]
        )
        
        # 최종 FC 레이어: 시간축 변환 (seq_len → pred_len)
        # 논문: "output of each layer is fed through skip connections 
        #        into a fully connected layer to produce the predictive output"
        self.linear = nn.Linear(input_shape[0], pred_len)
        
    def forward(self, x):
        """
        순전파: 입력 윈도우 → 미래 예측값.
        
        논문의 수식: ŷ_t = f(X_{t-1})
        
        Args:
            x: (batch, seq_len, num_features) = (batch, τ, N)
               입력 슬라이딩 윈도우 X_{t-1} = {x_{t-τ}, ..., x_{t-1}}
        
        Returns:
            x: (batch, pred_len, num_features) = (batch, 1, N)
               예측값 ŷ_t
        """
        # 1. RevIN 정규화: 각 시퀀스의 평균/표준편차로 정규화
        x = self.rev_norm(x, 'norm')
        
        # 2. L개의 Mixer Predictor Layer 통과
        # 각 레이어에서 Temporal Mixing + Feature Mixing 수행
        for res_block in self.res_blocks:
            x = res_block(x)

        # 3. 타겟 변수 선택 (M 모드에서는 전체 변수)
        if self.target_slice:
            x = x[:, :, self.target_slice]

        # 4. FC 레이어로 시간축 변환: seq_len → pred_len
        # (batch, τ, N) → transpose → (batch, N, τ) → Linear → (batch, N, pred_len)
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        # (batch, N, pred_len) → transpose → (batch, pred_len, N)
        x = torch.transpose(x, 1, 2)
        
        # 5. RevIN 역정규화: 원래 스케일로 복원
        x = self.rev_norm(x, 'denorm', self.target_slice)
        
        return x
