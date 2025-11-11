# future_prediction.py

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os

# GPU 사용 가능 여부 확인 (model_training.py와 동일하게 설정)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# 🚨 모델 클래스는 model_training.py에서 복사해와야 합니다.
# ---------------------------------------------------------------------

class LSTMModel(nn.Module):
    """LSTM 모델 정의 (model_training.py에서 복사)"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_sizes[2], 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x = x[:, -1, :] # 마지막 시퀀스만 사용
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# ---------------------------------------------------------------------

def create_sequences(X, time_steps):
    """시계열 시퀀스 생성 (y가 없는 예측 모드)"""
    X_seq = []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
    return np.array(X_seq)


def load_model_and_config(model_dir='models'):
    """저장된 모델, 스케일러, 설정을 로드"""
    
    # 설정 로드
    with open(f'{model_dir}/config.pkl', 'rb') as f:
        config = pickle.load(f)
        
    # 스케일러 로드
    with open(f'{model_dir}/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
        
    with open(f'{model_dir}/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    # 모델 초기화 및 가중치 로드
    input_size = len(config['feature_columns'])
    model = LSTMModel(input_size=input_size, output_size=len(config['target_columns']))
    model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ 모델 및 설정 로드 완료. Time Steps: {config['time_steps']}")
    return model, scaler_X, scaler_y, config


def run_future_prediction(future_data_path, models_dir='models'):
    """미래 데이터 예측 실행"""
    
    # 1. 모델 및 설정 로드
    model, scaler_X, scaler_y, config = load_model_and_config(models_dir)
    time_steps = config['time_steps']
    feature_columns = config['feature_columns']
    
    # 2. 예측 대상 데이터 로드 및 전처리
    # ⚠️ future_data_path는 10월 27일 데이터뿐 아니라, 
    #    TIME_STEPS만큼의 과거(10월 26일) 데이터가 포함된 파일이어야 합니다.
    data_full = pd.read_csv(future_data_path, parse_dates=['Date&Time'])
    
    # 예측에 필요한 특징 컬럼만 선택
    X_future_data = data_full[feature_columns].copy()
    
    # 3. 데이터 스케일링 (학습된 scaler_X 사용)
    X_future_scaled = scaler_X.transform(X_future_data)
    
    # 4. 예측 시퀀스 생성
    # 10월 27일 00:00 예측을 위해서는 10월 26일 01:00부터 23:00까지의 데이터가 필요합니다.
    # 즉, 예측하고 싶은 날짜의 데이터 (10월 27일 전체) + TIME_STEPS 만큼의 과거 데이터가 필요합니다.
    X_future_seq = create_sequences(X_future_scaled, time_steps)
    
    if len(X_future_seq) == 0:
        print(f"❌ 시퀀스 생성 실패: 데이터 길이가 Time Steps ({time_steps})보다 짧습니다.")
        return None
        
    print(f"미래 예측 시퀀스 수: {len(X_future_seq)}개")
    
    # 5. 예측 실행
    X_tensor = torch.FloatTensor(X_future_seq).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
        
    # 6. 스케일 복원
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    
    # 7. 🚨 수정: 10월 27일 타임스탬프만 추출 및 결과 정리
    # ---------------------------------------------------------------------
    
    # 가정: model_training.py에서 TRAIN_END_DATE가 10월 26일 23:00였으므로,
    # y_pred_original의 첫 번째 요소는 10월 27일 00:00의 예측값입니다.
    
    # 1. 예측된 10월 27일 데이터 (24개)를 슬라이싱합니다.
    y_pred_27th = y_pred_original[:24] # 첫 24개 레코드가 10월 27일 예측값입니다.
    
    # 2. data_full에서 10월 27일 00:00에 해당하는 인덱스를 찾습니다. (오류 유발 코드 대체)
    start_time_target = pd.to_datetime('2025-10-27 00:00:00')
    
    # ⚠️ 수정된 방법: 인덱스를 찾는 대신, 10월 27일 00:00 이후의 레코드를 필터링합니다.
    full_timestamps = data_full['Date&Time']
    
    # 10월 27일 00:00 이후 24개 레코드를 타임스탬프로 사용합니다.
    future_timestamps_27th = full_timestamps[full_timestamps >= start_time_target].iloc[:24].values
    
    # 3. 결과 DataFrame 생성 (24개 예측값 + 24개 타임스탬프)
    results_df = pd.DataFrame(y_pred_27th, columns=[f'Predicted_{col}' for col in config['target_columns']])
    results_df['Date&Time'] = future_timestamps_27th
    
    # 8. 결과 저장
    output_path = 'output/prediction_20251027_future.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ 미래 예측 완료! 저장 위치: {output_path}")
    print(results_df.head())
    
    return results_df


if __name__ == '__main__':
    # ⚠️ 주의: 이 파일 경로는 10월 27일의 미기후를 예측하는 데 필요한 
    #         과거 TIME_STEPS 기간(10월 26일)과 10월 27일의 모든 특징(ASOS)을 포함해야 합니다.
    #         일반적으로 model_training.py에서 사용한 'preprocessed_data.csv' 파일 전체를 사용하면 됩니다.
    FUTURE_DATA_PATH = 'output/preprocessed_data.csv' 
    
    if not os.path.exists(FUTURE_DATA_PATH):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {FUTURE_DATA_PATH}")
    elif not os.path.exists('models/best_model.pth'):
        print("❌ 학습된 모델 (models/best_model.pth)을 찾을 수 없습니다. model_training.py를 먼저 실행하세요.")
    else:
        run_future_prediction(FUTURE_DATA_PATH)