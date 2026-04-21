"""
=============================================================================
test.py - GCAD의 핵심: Granger 인과관계 추출 및 이상 탐지
=============================================================================
역할: 
  1. save_train_mean_causal(): 학습 데이터에서 정상 인과 패턴(Ā_norm) 추출
  2. test(): 테스트 데이터에서 인과 패턴 추출 → 정상 패턴과 비교 → 이상 점수 계산

연결: main.py에서 학습 완료 후 호출

논문 연결 (전체 파이프라인):
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1. Prediction-based Gradient Generator (models/tsmixer.py)      │
  │    → 학습된 예측기에서 채널별 그래디언트 추출                      │
  │                                                                  │
  │ 2. Granger Causality Discovery (이 파일)                         │
  │    → 그래디언트를 시간축으로 적분하여 인과행렬 A 계산              │
  │    → 논문 Equation 5: a_{i,j} = ∫|∂L_{t,j}/∂x_{φ,i}| dφ       │
  │                                                                  │
  │ 3. Causality Graph Sparsification (이 파일)                      │
  │    → 대칭 제거: Ã = max(0, A - A^T)  (논문 Equation 6)          │
  │    → 임계값 적용: 작은 값을 0으로                                 │
  │                                                                  │
  │ 4. Causal Deviation Scoring (이 파일)                            │
  │    → 테스트 인과행렬과 정상 패턴의 편차 계산                      │
  │    → 논문 Equation 10: Sc_i = Σ|Ã_test,i - Ā_norm|/(Ā_norm+ε) │
  └──────────────────────────────────────────────────────────────────┘
=============================================================================
"""

import argparse
import os
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def simple_moving_average(arr, label, window_size):
    """
    단순 이동 평균 (SMA)으로 이상 점수를 스무딩.
    
    이상 점수의 노이즈를 줄여 더 안정적인 탐지를 위해 사용.
    
    Args:
        arr: 이상 점수 배열
        label: 대응하는 라벨 배열
        window_size: 이동 평균 윈도우 크기 (홀수여야 함)
    
    Returns:
        [스무딩된 점수, 대응하는 라벨] (양쪽 끝이 잘림)
    """
    if len(arr) != len(label):
        print("len(score) != len(label)")
        return None
    
    moving_averages = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    
    # 양쪽 끝에서 (window_size-1)/2 만큼 잘림
    n = int((window_size - 1) / 2)
    return [moving_averages, label[n:len(arr) - n]]


def get_err_norm_parms(model, save_path, dataloader, device, parms_path, sample_p=0.1):
    """
    [사용되지 않는 함수] 학습 데이터의 그래디언트 통계(평균/표준편차) 계산.
    
    초기 버전에서 사용되었을 수 있으나, 현재 파이프라인에서는 
    save_train_mean_causal()이 대신 사용됨.
    """
    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)

    model.eval()
    
    test_mloss = torch.zeros(1, device=device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    all_loss = []
    first_tag = 0

    for i, (batch_x, batch_y) in pbar:
        sample = random.random()
        if (sample <= sample_p) or (first_tag == 0):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x.requires_grad = True
            outputs = model(batch_x).float().to(device) 

            loss = criterion(outputs, batch_y)
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)

            Grad = torch.autograd.grad(loss, batch_x, allow_unused=True, 
                                       create_graph=False, retain_graph=False)
            Grad = Grad[0]
            Grad = torch.abs(Grad)

            if first_tag == 0:
                first_tag = 1
                input_grad = Grad
            else:
                input_grad = torch.cat((input_grad, Grad), dim=0)

    temp_shape = input_grad.shape
    input_grad_reshape = torch.reshape(input_grad, ((-1, temp_shape[-1])))
    err_norm_mean = torch.mean(input_grad_reshape, dim=0, keepdim=False)
    err_norm_std = torch.std(input_grad_reshape, dim=0, keepdim=False)
    
    err_norm_mean = err_norm_mean.data.cpu().numpy()
    err_norm_std = err_norm_std.data.cpu().numpy()
    
    df = pd.DataFrame({
        'mean': err_norm_mean,
        'std': err_norm_std
    })
    df.to_csv(parms_path)   



def save_train_mean_causal(model, save_path, dataloader, device, parms_path, sparse_th, sample_p=0.01):
    """
    ★★★ GCAD 핵심 함수 1: 학습 데이터에서 정상 인과 패턴(Ā_norm) 추출 ★★★
    
    논문 연결:
      - Section: Causal Deviation Scoring, Equation 7-8
      - "After model training, we sample the training set windows using a 
         Bernoulli distribution and calculate the Granger causality graphs 
         for these samples."
      - "We use the mean matrix of the graph matrix sequence to represent 
         the typical normal causal pattern."
    
    전체 흐름:
      1. 학습된 모델 로드
      2. 학습 데이터를 Bernoulli 샘플링 (sample_p 확률로)
      3. 각 샘플에 대해 채널별 그래디언트 계산 → 인과행렬 생성
      4. 대칭 제거 (Sparsification) → 단방향 인과관계만 보존
      5. 임계값 적용 → 노이즈 제거
      6. 전체 샘플의 평균 → 정상 인과 패턴 Ā_norm
      7. CSV 파일로 저장
    
    Args:
        model: TSMixerRevIN 모델
        save_path: 학습된 모델 가중치 경로
        dataloader: 학습 데이터 DataLoader
        device: 연산 장치 (cuda/cpu)
        parms_path: 정상 인과 패턴을 저장할 CSV 경로
        sparse_th: 희소화 임계값 h (논문의 sparsification threshold)
        sample_p: Bernoulli 샘플링 확률 p (논문 Equation 7의 p)
    """
    # ===== 1. 학습된 모델 로드 =====
    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)
    model.eval()
    
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    first_tag = 0
    
    print("sampling causal matrix on train set with rate", sample_p)

    # ===== 2. 학습 데이터 순회 + Bernoulli 샘플링 =====
    for i, (batch_x, batch_y) in pbar:
        """
        batch_x: (batch, seq_len, N) - 입력 윈도우 X_{t-1}
        batch_y: (batch, pred_len, N) - 타겟 y_t
        """
        
        # Bernoulli 샘플링: sample_p 확률로 이 배치를 사용
        # 논문 Eq.7: b_i ~ Bernoulli(p)
        sample = random.random()
        if (sample <= sample_p) or (first_tag == 0):
            
            model.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # ===== 3. 채널별 그래디언트 계산 (Channel-separated Gradient) =====
            # 논문 Section: Prediction-based Gradient Generator
            # "We propose a channel-separated Error detector to generate channel loss"
            
            # 입력에 대한 그래디언트를 계산하기 위해 requires_grad 설정
            batch_x.requires_grad = True
            outputs = model(batch_x).float().to(device)  # ŷ_t = f(X_{t-1})
            
            # 각 출력 채널(변수) j에 대해 개별적으로 역전파
            for features in range(outputs.shape[-1]):
                """
                ★ 핵심 아이디어 ★
                논문 Equation 1: L_{t,j} = (ŷ_{t,j} - y_{t,j})²
                
                각 변수 j의 예측 오차를 개별적으로 계산하고,
                이 오차를 입력 X에 대해 역전파하여 그래디언트를 얻음.
                
                이 그래디언트 ∂L_{t,j}/∂x_{t',i}는:
                "변수 i의 시점 t'에서의 값이 변수 j의 예측에 얼마나 영향을 미치는가"
                를 나타냄 → 이것이 Granger 인과관계의 핵심!
                """
                model.zero_grad()
                
                # 채널 j의 예측 오차 계산 (논문 Eq.1)
                loss_i = criterion(outputs[:, :, features], batch_y[:, :, features])
                loss_i.requires_grad_(True)
                # 역전파: ∂L_{t,j}/∂X_t 계산
                loss_i.backward(retain_graph=True)

                # 입력에 대한 그래디언트 추출
                # Grad_i shape: (batch, seq_len, N)
                # Grad_i[b, t', i] = |∂L_{t,j}/∂x_{t',i}| for batch b
                Grad_i = batch_x.grad
                Grad_i = torch.abs(Grad_i)  # 절대값 (논문 Eq.5의 |·|)
                
                # 다음 채널을 위해 그래디언트 초기화
                batch_x.grad = None  
                
                # 채널별 그래디언트를 쌓아서 인과행렬 구성
                if features == 0:
                    # 첫 번째 채널: 새 차원 추가
                    grad_causal_mat = torch.unsqueeze(Grad_i, dim=3)
                else:
                    # 이후 채널: 마지막 차원에 연결
                    # grad_causal_mat shape: (batch, seq_len, N_input, N_output)
                    grad_causal_mat = torch.cat(
                        [grad_causal_mat, torch.unsqueeze(Grad_i, dim=3)], dim=3
                    )
            
            # 배치들을 누적
            if first_tag == 0:
                input_grad_causal_map = grad_causal_mat
                first_tag = 1
            else:
                # shape: (누적_samples, seq_len, N_input, N_output)
                input_grad_causal_map = torch.cat(
                    [input_grad_causal_map, grad_causal_mat], dim=0
                )

    # ===== 4. 시간축 적분 (Granger Causality Discovery) =====
    """
    논문 Equation 5: a_{i,j} = ∫_{t-τ}^{t-1} |∂L_{t,j}/∂x_{φ,i}| P(x_{φ,i}) dx_{φ,i}
    
    구현에서는 시간축(dim=1)에 대한 평균으로 근사:
      input_grad_causal_map: (samples, seq_len, N_input, N_output)
      → mean(dim=1) → (samples, N_input, N_output)
    
    결과: input_grad_causal_mat[s, i, j] = 샘플 s에서 변수 i가 변수 j에 미치는 인과 효과
    """
    input_grad_causal_mat = torch.mean(input_grad_causal_map, dim=1)
    
    # ===== 5. 인과 그래프 희소화 (Causality Graph Sparsification) =====
    """
    논문 Equation 6: Ã_{i,j} = max(0, A_{i,j} - A^T_{i,j}), i ≠ j
    
    목적: 양방향 대칭 관계(유사성)를 제거하고 단방향 인과관계만 보존
    
    구현 방법:
      1. 상삼각 행렬 추출
      2. 하삼각 행렬을 전치하여 상삼각 위치로 이동
      3. 차이 계산: 상삼각 - 하삼각^T
      4. 양수 부분 → 상삼각에 배치 (i→j 방향 인과)
      5. 음수 부분 → 하삼각에 배치 (j→i 방향 인과)
    """
    # 상삼각 부분 (대각선 포함)
    upper_triangle = torch.triu(input_grad_causal_mat, diagonal=0)
    # 하삼각 부분을 전치 (대각선 미포함)
    lower_triangle_transposed = torch.tril(input_grad_causal_mat, diagonal=-1).transpose(1, 2)
    # 차이 계산 (상삼각 부분만)
    result = torch.triu(upper_triangle - lower_triangle_transposed, diagonal=0)

    # 양수 부분: i→j 방향의 순수 인과 효과 (상삼각)
    result_upper = torch.where(result < 0, torch.zeros_like(result).to(device), result)
    # 음수 부분: j→i 방향의 순수 인과 효과 (하삼각으로 전치)
    result_lower = torch.where(
        result < 0, torch.abs(result).to(device), torch.zeros_like(result).to(device)
    ).transpose(1, 2)
    # 합치기: 완전한 희소화된 인과행렬
    input_grad_causal_mat = result_upper + result_lower
    
    # ===== 6. 임계값 적용 =====
    """
    논문: "We set a sparsity threshold h, setting causality effect values 
           below this threshold in the causality graph matrix to zero."
    "This is because insignificant causal relationships may be caused by noise"
    """
    zero = torch.zeros_like(input_grad_causal_mat).to(device)
    input_grad_causal_mat = torch.where(
        input_grad_causal_mat < sparse_th, zero, input_grad_causal_mat
    )

    # ===== 7. 정상 인과 패턴 계산 (평균) =====
    """
    논문 Equation 8: Ā_norm = (1/n) Σ Ã_{norm,i}
    
    모든 샘플의 희소화된 인과행렬의 평균 → 정상 인과 패턴
    shape: (N_input, N_output) = (N, N)
    """
    input_grad_causal_map = torch.mean(input_grad_causal_mat, dim=0)
    
    # ===== 8. CSV 파일로 저장 =====
    input_grad_causal_map = input_grad_causal_map.data.cpu().numpy()
    df = pd.DataFrame(input_grad_causal_map)
    df.to_csv(parms_path, index=False, header=False)



def test(model, save_path, dataloader, device, parms_path, sparse_th):
    """
    ★★★ GCAD 핵심 함수 2: 테스트 데이터에서 이상 탐지 수행 ★★★
    (메모리 최적화 버전: 배치마다 바로 점수 계산 → GPU 메모리 절약)
    
    원본과 동일한 로직:
      그래디언트 계산 → 시간축 적분 → 희소화 → 임계값 → 편차 점수
    차이점:
      원본은 모든 샘플의 인과행렬을 GPU에 쌓은 뒤 한번에 처리했지만,
      이 버전은 배치마다 바로 점수를 계산하고 CPU로 옮겨서 메모리를 절약함.
      수학적으로 완전히 동일한 결과를 냄.
    
    Args:
        model: TSMixerRevIN 모델
        save_path: 학습된 모델 가중치 경로
        dataloader: 테스트 데이터 DataLoader
        device: 연산 장치
        parms_path: 정상 인과 패턴 CSV 경로
        sparse_th: 희소화 임계값 h
    
    Returns:
        eva_list: [auc_roc, auc_prc, f1, precision, recall, f1_pa]
    """
    # ===== 1. 학습된 모델 로드 =====
    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)
    model.eval()
    
    criterion = torch.nn.MSELoss(reduction='sum').to(device)

    # ===== 정상 인과 패턴 미리 로드 =====
    standard_causal_mat_df = pd.read_csv(parms_path, index_col=None, header=None)
    standard_causal_mat = torch.tensor(np.array(standard_causal_mat_df)).to(device)
    standard_causal_mat = standard_causal_mat + 1e-4  # ε 추가

    print(('\n' + '%-10s' * 1) % ('Test loss'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    # CPU에 점수와 라벨을 누적 (GPU 메모리 사용 안 함)
    all_err_scores = []
    all_labels = []

    # ===== 2. 테스트 데이터 순회: 배치마다 바로 점수 계산 =====
    for i, (batch_x, batch_y, batch_labels) in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_labels = batch_labels.to(device)
        
        # 윈도우 내 라벨의 최대값 → 윈도우 단위 라벨
        label, _ = torch.max(batch_labels, dim=1)

        batch_x.requires_grad = True
        outputs = model(batch_x).float().to(device)
        
        # ===== 채널별 그래디언트 계산 (원본과 동일) =====
        for features in range(outputs.shape[-1]):
            model.zero_grad()
            Grad_i = torch.zeros_like(batch_x)
                
            loss_i = criterion(outputs[:, :, features], batch_y[:, :, features])
            loss_i.requires_grad_(True)
            loss_i.backward(retain_graph=True)
            
            Grad_i = batch_x.grad
            Grad_i = torch.abs(Grad_i)
            batch_x.grad = None 
            
            if features == 0:
                grad_causal_mat = torch.unsqueeze(Grad_i, dim=3)
            else:
                grad_causal_mat = torch.cat(
                    [grad_causal_mat, torch.unsqueeze(Grad_i, dim=3)], dim=3
                )
        
        # ===== 이 배치에 대해 바로 처리 (원본과 동일한 연산) =====
        
        # 시간축 적분 (논문 Eq.5)
        batch_causal_mat = torch.mean(grad_causal_mat, dim=1)  # (batch, N, N)
        
        # 희소화 (논문 Eq.6)
        upper_triangle = torch.triu(batch_causal_mat, diagonal=0)
        lower_triangle_transposed = torch.tril(batch_causal_mat, diagonal=-1).transpose(1, 2)
        result = torch.triu(upper_triangle - lower_triangle_transposed, diagonal=0)
        result_upper = torch.where(result < 0, torch.zeros_like(result).to(device), result)
        result_lower = torch.where(
            result < 0, torch.abs(result).to(device), torch.zeros_like(result).to(device)
        ).transpose(1, 2)
        batch_causal_mat = result_upper + result_lower
        
        # 임계값 적용
        zero = torch.zeros_like(batch_causal_mat).to(device)
        batch_causal_mat = torch.where(batch_causal_mat < sparse_th, zero, batch_causal_mat)
        
        # 편차 점수 계산 (논문 Eq.10)
        for s in range(batch_causal_mat.shape[0]):
            temp_error = torch.abs(batch_causal_mat[s, :, :] - standard_causal_mat)
            temp_error = torch.div(temp_error, standard_causal_mat)
            temp_error = torch.mean(temp_error)
            all_err_scores.append(temp_error.item())  # CPU로 바로 저장
        
        # 라벨도 CPU로 저장
        all_labels.extend(label.cpu().numpy().tolist())
        
        # 이 배치의 GPU 메모리 해제
        del grad_causal_mat, batch_causal_mat, outputs
        torch.cuda.empty_cache()

    # ===== NumPy 변환 =====
    err_score = np.array(all_err_scores)
    test_labels = np.array(all_labels)
    
    # ===== SMA 스무딩 =====
    smoothed = simple_moving_average(err_score, test_labels, window_size=3)
    err_score = smoothed[0]
    test_labels = smoothed[1]
    
    # ===== 평가 지표 계산 =====
    auc_score = roc_auc_score(test_labels, err_score)
    print("ROC_score:", auc_score)
    
    precision, recall, thresholds = precision_recall_curve(test_labels, err_score)
    auc_precision_recall = auc(recall, precision)
    print("PRC_score:", auc_precision_recall)
    
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    f1 = np.max(f1_scores)
    pre = precision[np.argmax(f1_scores)]
    rec = recall[np.argmax(f1_scores)]
    print('Best F1-Score: ', f1)
    print('Precision: ', pre)
    print('Recall: ', rec)
    
    # ===== Point Adjustment =====
    th_pa = thresholds[np.argmax(f1_scores)]
    pred = np.where(err_score >= th_pa, 1, 0).astype(int)
    gt = test_labels.astype(int)
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    
    accuracy_pa = accuracy_score(gt, pred)
    precision_pa, recall_pa, f1_pa, support = precision_recall_fscore_support(
        gt, pred, average='binary'
    )
    
    eva_list = [auc_score, auc_precision_recall, f1, pre, rec, f1_pa]
            
    return eva_list 
