#!/bin/bash
# =============================================================================
# run.sh - 각 데이터셋별 실행 명령어
# =============================================================================
# 각 데이터셋마다 최적화된 하이퍼파라미터가 다름.
# 
# 주요 하이퍼파라미터 설명:
#   --seq_len    : τ (최대 시간 지연) - Granger 인과관계의 시간 범위
#   --pred_len   : 예측 길이 (보통 1)
#   --pd_beta    : β (인과 편차와 시간 패턴 편차의 균형, 논문 Eq.12)
#                  S = Sc + β·St
#   --sample_p   : p (Bernoulli 샘플링 확률, 논문 Eq.7)
#   --sparse_th  : h (희소화 임계값, 논문 Section: Causality Graph Sparsification)
#   --test_stride: 테스트 시 윈도우 이동 간격 (계산 효율성)
#   --n_block    : L (Mixer Layer 수)
#   --ff_dim     : Feature Mixing 은닉 차원
# =============================================================================

# SMD 데이터셋 (38채널, 서버 머신 데이터)
# τ=30, β=0, h=0.005
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --data ./dataset/smd --name smd --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001

# SWaT 데이터셋 (51채널, 수처리 시스템)
# τ=5 (짧은 시간 지연), β=0.5 (시간 패턴도 활용), h=0.008
python main.py --seq_len 5 --pred_len 1 --pd_beta 0.5 --sample_p 0.2 --sparse_th 0.008 --test_stride 5 --data ./dataset/swat --name swat --n_block 6 --ff_dim 2048 --dropout 0 --learning_rate 0.0001

# SMAP 데이터셋 (25채널, NASA 우주선 데이터)
# τ=70 (긴 시간 지연), β=1 (시간 패턴 강조), h=0.008
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --data ./dataset/smap --name smap --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001

# PSM 데이터셋 (25채널, 서버 머신 데이터)
# τ=30, pred_len=5 (5시점 예측), β=0.5, h=0.005
python main.py --seq_len 30 --pred_len 5 --pd_beta 0.5 --sample_p 0.1 --sparse_th 0.005 --test_stride 10 --data ./dataset/psm --name psm --n_block 2 --ff_dim 128 --dropout 0 --learning_rate 0.0001

# MSL 데이터셋 (55채널, NASA 화성 탐사 로버 데이터)
# τ=30, β=0, h=0.002
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --data ./dataset/msl --name msl --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001
