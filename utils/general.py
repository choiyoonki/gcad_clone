"""
=============================================================================
utils/general.py - 유틸리티: 시드 설정
=============================================================================
역할: 실험 재현성을 위한 랜덤 시드 설정
연결: main.py에서 학습 시작 전에 호출됨

논문 연결:
  - "All experiments were conducted 10 times and the average results were reported."
    (논문 Section: Experimental Setup)
  - 10번 반복 실험의 각 실행마다 다른 랜덤 시드를 사용하여 결과의 분산을 측정
=============================================================================
"""

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=0):
    """
    랜덤 시드를 설정하여 실험의 재현성을 관리하는 함수.
    
    주의: 이 구현에서는 입력된 seed 값을 무시하고 0~1000 사이의 랜덤 시드를 생성함.
    이는 논문에서 10번 반복 실험을 수행할 때 매번 다른 시드로 실험하기 위한 설계.
    
    Args:
        seed (int): 사용되지 않음 (내부에서 랜덤 생성)
    """
    # 0~1000 사이의 랜덤 시드 생성 (매 실행마다 다른 시드)
    seed = random.randint(0, 1000) 
    
    # Python 기본 random 모듈 시드 고정
    random.seed(seed)
    
    # NumPy 시드 고정
    np.random.seed(seed)
    
    # PyTorch CPU 시드 고정
    torch.manual_seed(seed)
    
    # cuDNN 설정: benchmark=False, deterministic=True로 재현성 확보
    # benchmark=False: 입력 크기가 변할 때 최적 알고리즘 자동 탐색 비활성화
    # deterministic=True: 동일 입력에 대해 항상 같은 결과 보장
    cudnn.benchmark, cudnn.deterministic = (False, True)
    
    # PyTorch GPU 시드 고정 (단일 GPU)
    torch.cuda.manual_seed(seed)
    
    # PyTorch GPU 시드 고정 (멀티 GPU)
    torch.cuda.manual_seed_all(seed)
