"""
=============================================================================
utils/dataloader.py - 데이터 로딩 및 전처리
=============================================================================
역할: 시계열 데이터를 읽어서 학습/검증/테스트 DataLoader를 생성
연결: main.py에서 SwatDataLoader_AD를 사용하여 데이터를 로드

논문 연결:
  - "Each dataset consists of two parts: unlabeled normal operation data 
     and labeled data containing some anomalies." (논문 Section: Implementation Details)
  - "We use 80% of the normal data for training, and the remaining 20% 
     is used for the validation set." (논문 Section: Implementation Details)
  - "Testing is conducted on the data containing anomalies."
  
데이터 구조:
  - train.csv: 정상 운영 데이터 (라벨 없음, 마지막 열은 라벨이지만 모두 0)
  - test.csv: 이상 포함 데이터 (마지막 열이 이상 라벨: 0=정상, 1=이상)
  
슬라이딩 윈도우:
  - 논문의 입력: {x_{t-τ}, ..., x_{t-1}} → 길이 τ(=seq_len)의 윈도우
  - 논문의 출력: x_t의 예측값 → pred_len 길이의 미래값
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SwatDataLoader_AD:
    """
    이상 탐지용 데이터 로더 (SWaT, SMD, MSL, SMAP, PSM 등에 사용).
    
    논문의 Problem Statement와 대응:
      - 입력: 슬라이딩 윈도우 {x_{t-τ}, ..., x_{t-1}, x_t}
      - 출력: 해당 윈도우가 이상인지 여부 (Boolean)
    
    데이터 흐름:
      1. CSV 파일 읽기 → train.csv (정상 데이터), test.csv (이상 포함 데이터)
      2. 80/20 분할 → 학습셋 / 검증셋
      3. StandardScaler로 정규화 (학습셋 기준)
      4. 슬라이딩 윈도우 방식으로 DataLoader 생성
    """

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT', stride=1
        ):
        """
        Args:
            data (str): 데이터셋 폴더 경로 (예: './datasets/smd')
            batch_size (int): 배치 크기
            seq_len (int): 입력 시퀀스 길이 = 논문의 τ (최대 시간 지연)
                           이 값이 Granger 인과관계에서 고려하는 최대 시간 지연
            pred_len (int): 예측 길이 (보통 1, 다음 시점 예측)
            feature_type (str): 'M'=다변량→다변량, 'S'=단변량→단변량, 'MS'=다변량→단변량
            target (str): MS 모드에서 타겟 변수명
            stride (int): 테스트 시 슬라이딩 윈도우 이동 간격
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len          # 논문의 τ (최대 시간 지연)
        self.pred_len = pred_len        # 예측 길이 (보통 1)
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)  # 기본: 모든 변수를 타겟으로 사용
        self.stride = stride

        self._read_data()

    def _read_data(self):
        """
        CSV 파일을 읽고 학습/검증/테스트 데이터로 분할.
        
        데이터 분할 방식 (논문 Section: Implementation Details):
          - 학습: 정상 데이터의 80%
          - 검증: 정상 데이터의 나머지 20%
          - 테스트: 이상이 포함된 별도 데이터
        """
        # ===== 1. CSV 파일 읽기 =====
        # train.csv: 정상 운영 데이터 (마지막 열은 라벨 컬럼)
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        # test.csv: 이상 포함 데이터 (마지막 열이 이상 라벨)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        # 학습 데이터에서 마지막 열(라벨) 제거 → 센서 값만 사용
        df = df_raw.iloc[:,:-1]
        # 테스트 데이터의 라벨 분리
        df_test_labels = df_test_raw.iloc[:,-1]   # 이상 라벨 (0 또는 1)
        df_test_value = df_test_raw.iloc[:,:-1]    # 센서 값

        # ===== 2. 학습/검증 분할 (80/20) =====
        n = len(df)
        train_end = int(n * 0.8)    # 80% 지점
        val_end = n                  # 나머지 20%

        train_df = df[:train_end]
        # 검증셋은 seq_len만큼 앞에서 겹치게 시작 (슬라이딩 윈도우를 위해)
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        # ===== 3. 표준화 (StandardScaler) =====
        # 학습 데이터의 평균/표준편차로 정규화 (데이터 누수 방지)
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            """DataFrame을 StandardScaler로 변환"""
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)

        self.test_labels = df_test_labels

        # 변수(센서/채널) 수 = 논문의 N
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, stride=1):
        """
        슬라이딩 윈도우 방식으로 DataLoader를 생성.
        
        슬라이딩 윈도우 (논문 Section: Problem Statement):
          - 입력 X_{t-1} = {x_{t-τ}, x_{t-τ+1}, ..., x_{t-1}} → shape: (seq_len, N)
          - 타겟 y_t = x_t → shape: (pred_len, N)
        
        Args:
            data: 정규화된 DataFrame
            shuffle: 데이터 셔플 여부 (학습=True, 검증/테스트=False)
            testing: 테스트 모드 여부 (True면 라벨도 반환)
            test_labels: 테스트 라벨
            stride: 윈도우 이동 간격 (테스트 시 계산량 줄이기 위해 사용)
        """
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            # 테스트: stride 간격으로 윈도우 생성 + 라벨 포함
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, 
                                  testing=True, test_labels=test_labels),
                    range(0, len(data_x) - self.seq_len - self.pred_len + 1, stride)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )
        else:
            # 학습/검증: 모든 윈도우 사용
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )

    def get_train(self, shuffle=True):
        """학습 DataLoader 반환"""
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        """검증 DataLoader 반환"""
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        """테스트 DataLoader 반환 (라벨 포함)"""
        return self._make_dataset(self.test_df, shuffle=False, testing=True, 
                                  test_labels=self.test_labels, stride=self.stride)


class smdDataLoader_AD:
    """
    SMD 데이터셋 전용 로더 (SwatDataLoader_AD와 거의 동일).
    차이점: train.csv에 라벨 컬럼이 없어서 iloc[:,:-1] 처리가 다름.
    실제로는 SwatDataLoader_AD로 통합 사용 가능.
    """

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """데이터 읽기 및 분할 (SwatDataLoader_AD와 유사)"""
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        df = df_raw  # SMD는 train.csv에 라벨 컬럼이 없음
        df_test_labels = df_test_raw.iloc[:,-1]
        df_test_value = df_test_raw.iloc[:,:-1]

        n = len(df)
        train_end = int(n * 0.8)
        val_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)

        self.test_labels = df_test_labels
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, train_average=False):
        data = np.array(data, dtype=np.float32)
        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, 
                                  testing=True, test_labels=test_labels),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True
            )
        else:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False, testing=True, 
                                  test_labels=self.test_labels)


class CustomDataLoader:
    """
    일반 시계열 예측용 데이터 로더 (ETT 등 벤치마크 데이터셋용).
    GCAD 이상 탐지에서는 사용되지 않음. 참고용.
    """

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self._read_data()

    def _read_data(self):
        df_raw = pd.read_csv(self.data)
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        n = len(df)
        if self.data.stem.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.stem.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)
        return DataLoader(
            torch.utils.data.Subset(
                CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                range(len(data_x) - self.seq_len - self.pred_len + 1)
            ),
            batch_size=self.batch_size, 
            shuffle=shuffle
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)
    

class CustomDataset(Dataset):
    """
    슬라이딩 윈도우 기반 커스텀 Dataset.
    
    논문의 슬라이딩 윈도우 구현:
      - 입력 X_{t-1} = {x_{t-τ}, ..., x_{t-1}} → shape: (seq_len, num_features)
      - 타겟 ŷ_t = f(X_{t-1}) → shape: (pred_len, num_features)
      - 테스트 시 라벨도 함께 반환 (윈도우 내 이상 여부 판단용)
    """
    
    def __init__(self, data_x, data_y, seq_len, pred_len, testing=False, test_labels=None):
        """
        Args:
            data_x: 전체 입력 데이터 (전체 시계열)
            data_y: 전체 타겟 데이터 (target_slice 적용됨)
            seq_len: 입력 윈도우 길이 (= 논문의 τ)
            pred_len: 예측 길이
            testing: 테스트 모드 여부
            test_labels: 이상 라벨 (테스트 시에만 사용)
        """
        self.data_x = data_x
        self.data_y = data_y
        self.test_labels = test_labels
        self.seq_len = seq_len      # 논문의 τ (최대 시간 지연)
        self.pred_len = pred_len    # 예측 길이
        self.testing = testing

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        """
        idx 위치에서 슬라이딩 윈도우를 잘라서 반환.
        
        반환값:
          - data_x[idx : idx+seq_len]  → 입력 윈도우, shape: (seq_len, num_features)
          - data_y[idx+seq_len : idx+seq_len+pred_len] → 예측 타겟, shape: (pred_len, num_features)
          - (테스트 시) test_labels[idx : idx+seq_len+pred_len] → 윈도우 내 라벨
        
        예시 (seq_len=30, pred_len=1):
          idx=0 → 입력: x[0:30], 타겟: x[30:31]
          idx=1 → 입력: x[1:31], 타겟: x[31:32]
        """
        if self.testing:
            return (self.data_x[idx : idx + self.seq_len], 
                    self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len], 
                    self.test_labels[idx : idx + self.seq_len + self.pred_len])
        else:
            return (self.data_x[idx : idx + self.seq_len], 
                    self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len])
