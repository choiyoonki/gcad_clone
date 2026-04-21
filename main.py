"""
=============================================================================
main.py - GCAD 프레임워크의 메인 실행 파일
=============================================================================
역할: 전체 파이프라인을 조율하는 진입점
  1. 데이터 로드
  2. 모델 생성 및 학습 (예측 태스크)
  3. 정상 인과 패턴 추출 (save_train_mean_causal)
  4. 테스트 및 이상 탐지 (test)

논문 연결 (전체 흐름, Figure 1):
  ┌─────────────────────────────────────────────────────────────┐
  │  [Training Phase]                                           │
  │   데이터 로드 → TSMixer 예측기 학습 (MSE 손실)              │
  │   "During the training phase, the gradient generator is     │
  │    trained for the prediction task."                         │
  │                                                             │
  │  [Sampling Phase]                                           │
  │   학습 데이터 샘플링 → 그래디언트 추출 → 정상 인과 패턴 저장 │
  │   "We sample the training set windows using a Bernoulli     │
  │    distribution and calculate the Granger causality graphs" │
  │                                                             │
  │  [Testing Phase]                                            │
  │   테스트 데이터 → 그래디언트 추출 → 인과행렬 계산            │
  │   → 정상 패턴과 비교 → 이상 점수 산출                       │
  └─────────────────────────────────────────────────────────────┘

파일 간 연결 관계:
  main.py (이 파일)
    ├── utils/dataloader.py  → 데이터 로딩 (SwatDataLoader_AD)
    ├── utils/general.py     → 시드 설정 (set_seed)
    ├── models/tsmixer.py    → 예측 모델 (TSMixerRevIN)
    │   └── models/common.py → 모델 빌딩 블록 (ResBlock, RevIN)
    └── test.py              → 인과관계 추출 & 이상 탐지
        ├── save_train_mean_causal() → 정상 인과 패턴 추출
        └── test()                   → 테스트 & 평가
=============================================================================
"""

import argparse
import os
from pathlib import Path
import sys
import pandas as pd
from test import test, save_train_mean_causal

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 프로젝트 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT를 Python 경로에 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로로 변환

import torch
from tqdm import tqdm
from copy import deepcopy

from utils.general import set_seed
from utils.dataloader import SwatDataLoader_AD
from models.tsmixer import TSMixerRevIN


def main(args):
    """
    GCAD의 메인 실행 함수: 학습 → 정상 패턴 추출 → 테스트.
    
    Args:
        args: 커맨드라인 인자 (하이퍼파라미터)
    
    Returns:
        eva_list: [auc_roc, auc_prc, f1, precision, recall, f1_pa]
    """
    # ===== 장치 선택 및 시드 설정 =====
    device = torch.device(args.device)
    set_seed(args.seed)
    
    # ===== 1단계: 데이터 로드 =====
    """
    논문 Section: Implementation Details
    - train.csv → 정상 운영 데이터 (80% 학습, 20% 검증)
    - test.csv → 이상 포함 데이터
    """
    data_loader = SwatDataLoader_AD(
        args.data,           # 데이터셋 경로
        args.batch_size,     # 배치 크기
        args.seq_len,        # 입력 시퀀스 길이 = 논문의 τ (최대 시간 지연)
        args.pred_len,       # 예측 길이 (보통 1)
        args.feature_type,   # 'M': 다변량→다변량
        args.target,         # 타겟 변수 (M 모드에서는 사용 안 함)
    )

    train_data = data_loader.get_train()   # 학습 DataLoader
    val_data = data_loader.get_val()       # 검증 DataLoader
    test_data = data_loader.get_test()     # 테스트 DataLoader (라벨 포함)

    # ===== 2단계: 모델 생성 =====
    """
    논문 Section: Prediction-based Gradient Generator
    - TSMixer (Chen et al., 2023)를 예측기(= 그래디언트 생성기)로 사용
    - L개의 Mixer Predictor Layer로 구성
    - 입력: (batch, seq_len, N) → 출력: (batch, pred_len, N)
    """
    model = TSMixerRevIN(
        input_shape=(args.seq_len, data_loader.n_feature),  # (τ, N)
        pred_len=args.pred_len,          # 예측 길이
        dropout=args.dropout,            # 드롭아웃
        n_block=args.n_block,            # Mixer Layer 수 = 논문의 L
        ff_dim=args.ff_dim,              # Feature Mixing 은닉 차원
        target_slice=data_loader.target_slice,  # 타겟 변수 슬라이스
    ).to(device)

    # ===== 손실 함수 및 옵티마이저 설정 =====
    """
    논문: "During the training phase, the MSE loss is used to guide 
           the optimization of predictor parameters: L_train = MSE(ŷ_t, y_t)"
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = torch.tensor(float('inf'))

    # ===== 체크포인트 디렉토리 생성 =====
    save_directory = os.path.join(args.checkpoint_dir, args.name)

    if os.path.exists(save_directory):
        # 이미 존재하면 번호를 붙여서 새 디렉토리 생성 (예: smd2, smd3, ...)
        import glob
        import re
        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        save_directory = f"{path}{n}"

    os.makedirs(save_directory)

    # ===== 3단계: 모델 학습 (Training Phase) =====
    """
    논문 Section: Prediction-based Gradient Generator
    - 정상 데이터로 예측 태스크를 학습
    - "During the training phase, normal causal patterns are embedded 
       into the deep network."
    - MSE 손실로 최적화: L_train = MSE(ŷ_t, y_t)
    - Early Stopping으로 과적합 방지
    """
    for epoch in range(args.train_epochs):
    
        # ----- 학습 루프 -----
        model.train()
        train_mloss = torch.zeros(1, device=device)
        
        print(('\n' + '%-10s' * 2) % ('Epoch', 'Train loss'))
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        for i, (batch_x, batch_y) in pbar:
            """
            batch_x: (batch, seq_len, N) - 입력 윈도우 X_{t-1}
            batch_y: (batch, pred_len, N) - 타겟 y_t
            """
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 순전파: ŷ_t = f(X_{t-1})
            outputs = model(batch_x)
            
            optimizer.zero_grad()
            
            # MSE 손실 계산: L_train = MSE(ŷ_t, y_t)
            loss = criterion(outputs, batch_y)
            
            # 역전파 및 파라미터 업데이트
            loss.backward()
            optimizer.step()
            
            # 이동 평균 손실 계산 (로깅용)
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)

            pbar.set_description(('%-10s' * 1 + '%-10.4g' * 1) %
                                 (f'{epoch+1}/{args.train_epochs}', train_mloss))

        # ----- 검증 루프 -----
        model.eval()
        val_mloss = torch.zeros(1, device=device)
        
        print(('%-10s' * 2) % ('', 'Val loss'))
        pbar = tqdm(enumerate(val_data), total=len(val_data))
        
        with torch.no_grad(): 
            for i, (batch_x, batch_y) in pbar:
                """
                batch_x: (batch, seq_len, N) - 입력 윈도우
                batch_y: (batch, pred_len, N) - 타겟
                outputs: (batch, pred_len, N) - 예측값
                """
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.4g' * 1) %
                                     (f'', val_mloss))

            # ----- Early Stopping -----
            if val_mloss < best_loss:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))
                patience = 0
            else:
                patience += 1
                
            if (patience == args.patience) or (epoch >= args.train_epochs - 1):           
                break
            
    # ===== 4단계: 정상 인과 패턴 추출 (Sampling Phase) =====
    """
    논문 Section: Causal Deviation Scoring, Equation 7-8
    - 학습 데이터를 Bernoulli 샘플링하여 인과행렬 계산
    - 평균 인과행렬 = 정상 인과 패턴 Ā_norm
    - CSV 파일로 저장 (causal_parms.csv)
    """
    parms_path = os.path.join(save_directory, "causal_parms.csv")
    
    save_train_mean_causal(
        model,
        os.path.join(save_directory, "best.pt"),  # 최적 모델 가중치
        train_data,                                # 학습 데이터
        args.device,
        parms_path=parms_path,                     # 저장 경로
        sparse_th=args.sparse_th,                  # 희소화 임계값 h
        sample_p=args.sample_p                     # Bernoulli 샘플링 확률 p
    )

    # ===== 5단계: 테스트 및 이상 탐지 (Testing Phase) =====
    """
    논문 Section: Causal Deviation Scoring, Equation 9-12
    - 테스트 데이터의 각 윈도우에 대해 인과행렬 계산
    - 정상 패턴과의 편차 → 이상 점수
    - AUROC, AUPRC 등 평가 지표 계산
    """
    eva_list = test(
        model,
        os.path.join(save_directory, "best.pt"),  # 최적 모델 가중치
        test_data,                                 # 테스트 데이터
        args.device,
        parms_path=parms_path,                     # 정상 인과 패턴
        sparse_th=args.sparse_th                   # 희소화 임계값 h
    )
    return eva_list


def parse_args():
    """
    커맨드라인 인자 파싱.
    
    주요 하이퍼파라미터와 논문 연결:
      - seq_len: 논문의 τ (최대 시간 지연) → Granger 인과관계의 시간 범위
      - pred_len: 예측 길이 (보통 1)
      - sample_p: Bernoulli 샘플링 확률 p (논문 Eq.7)
      - sparse_th: 희소화 임계값 h (논문 Section: Causality Graph Sparsification)
      - n_block: Mixer Layer 수 L (논문 Section: Prediction-based Gradient Generator)
      - ff_dim: Feature Mixing 은닉 차원
    """
    parser = argparse.ArgumentParser()

    # ----- 기본 설정 -----
    parser.add_argument('--device', type=str, default='cuda:0', help='연산 장치')
    parser.add_argument('--train_epochs', type=int, default=100, help='최대 학습 에폭')
    parser.add_argument('--patience', type=int, default=2, help='Early Stopping 인내심')

    # ----- GCAD 핵심 하이퍼파라미터 -----
    parser.add_argument(
        '--seq_len', type=int, default=30,
        help='입력 시퀀스 길이 = 논문의 τ (최대 시간 지연). '
             'Granger 인과관계에서 고려하는 과거 시점의 범위.'
    )
    parser.add_argument(  
        '--pred_len', type=int, default=1,
        help='예측 길이. 보통 1 (다음 시점 예측).'
    )
    parser.add_argument(
        '--sample_p', type=float, default=0.2,
        help='학습셋 Bernoulli 샘플링 확률 p (논문 Eq.7). '
             '정상 인과 패턴 추출 시 사용할 학습 데이터 비율.'
    )
    parser.add_argument(
        '--sparse_th', type=float, default=0.005,
        help='희소화 임계값 h (논문 Section: Causality Graph Sparsification). '
             '이 값보다 작은 인과 효과는 노이즈로 간주하여 0으로 설정.'
    )
    parser.add_argument(
        '--test_stride', type=int, default=1,
        help='테스트 시 슬라이딩 윈도우 이동 간격. 클수록 빠르지만 해상도 감소.'
    )

    # ----- 데이터 설정 -----
    parser.add_argument('--data', type=str, default='./datasets/smd',
                        help='데이터셋 폴더 경로')
    parser.add_argument(
        '--feature_type', type=str, default='M',
        choices=['S', 'M', 'MS'],
        help='예측 태스크 유형. M: 다변량→다변량 (GCAD 기본값)'
    )
    parser.add_argument('--target', type=str, default='OT',
                        help='S/MS 모드에서 타겟 변수명')
    parser.add_argument('--checkpoint_dir', type=str, default=ROOT / './checkpoints',
                        help='모델 체크포인트 저장 경로')
    parser.add_argument('--name', type=str, default='smd',
                        help='실험 이름 (체크포인트 하위 폴더명)')

    # ----- 모델 하이퍼파라미터 -----
    parser.add_argument(
        '--n_block', type=int, default=3,
        help='Mixer Predictor Layer 수 = 논문의 L. 깊을수록 복잡한 관계 학습 가능.'
    )
    parser.add_argument(
        '--ff_dim', type=int, default=1024,
        help='Feature Mixing MLP의 은닉층 차원. 클수록 표현력 증가.'
    )
    parser.add_argument('--dropout', type=float, default=0, help='드롭아웃 비율')
    parser.add_argument('--norm_type', type=str, default='L', choices=['L', 'B'],
                        help='정규화 유형 (L: LayerNorm, B: BatchNorm)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'], help='활성화 함수')

    # ----- 최적화 설정 -----
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='학습률')
 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    메인 실행부: 10번 반복 실험 수행.
    
    논문: "All experiments were conducted 10 times and the average results 
           were reported."
    
    각 실행의 결과를 CSV로 저장하여 평균/분산 분석 가능.
    """
    eva_out = []
    for times in range(10):
        args = parse_args()
        # main() 반환값: [auc_roc, auc_prc, f1, precision, recall, f1_pa]
        temp = main(args)
        eva_out.append(temp)
        
        # 결과를 CSV로 저장 (매 실행마다 업데이트)
        df = pd.DataFrame(eva_out)
        df.columns = ['auc_roc', 'auc_prc', 'f1', 'pre', 'rec', 'f1_pa']
        df.to_csv('./result/smd.csv', index=False)
