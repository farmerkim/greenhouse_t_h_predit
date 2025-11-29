import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===== 설정 파라미터 =====
PREDICTION_HOURS = 6  # 3 또는 6시간 후 예측
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# ===== 커스텀 데이터셋 클래스 =====
class GreenhouseDataset(Dataset):
    def __init__(self, X_weather, X_greenhouse, y):
        self.X_weather = torch.FloatTensor(X_weather)
        self.X_greenhouse = torch.FloatTensor(X_greenhouse)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_weather[idx], self.X_greenhouse[idx], self.y[idx]

# ===== LSTM 모델 정의 =====
class GreenhousePredictionLSTM(nn.Module):
    """
    기상청 데이터 + 현재 온실 상태 -> N시간 후 온실 상태 예측
    """
    def __init__(self, weather_input_size, greenhouse_input_size, 
                 hidden_size, num_layers, output_size, dropout=0.2):
        super(GreenhousePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 기상 데이터 처리 LSTM
        self.weather_lstm = nn.LSTM(
            input_size=weather_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 온실 데이터 처리 레이어
        self.greenhouse_encoder = nn.Sequential(
            nn.Linear(greenhouse_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 통합 레이어
        combined_size = hidden_size + hidden_size // 2
        self.fc = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, weather_data, greenhouse_data):
        # 기상 데이터 LSTM 처리
        lstm_out, (h_n, c_n) = self.weather_lstm(weather_data)
        weather_features = lstm_out[:, -1, :]  # 마지막 타임스텝
        
        # 온실 데이터 인코딩
        greenhouse_features = self.greenhouse_encoder(greenhouse_data)
        
        # 특성 결합
        combined = torch.cat([weather_features, greenhouse_features], dim=1)
        
        # 최종 예측
        output = self.fc(combined)
        return output

# ===== 데이터 준비 함수 =====
def prepare_data(df, prediction_hours=6):
    """
    현재 시점의 (기상 + 온실) 데이터로 N시간 후 온실 상태 예측
    """
    print(f"\n데이터 준비: {prediction_hours}시간 후 예측 모델")
    
    # 컬럼 분류
    # 기상 데이터: outer_로 시작하거나 wind, rainfall, solar_rad, pressure
    weather_cols = [col for col in df.columns if col.startswith('outer_') or 
                    col in ['wind_speed', 'wind_dir', 'rainfall', 'solar_rad', 'pressure']]
    
    # 온실 현재 상태: inner_temp, inner_hum (lag 제외)
    greenhouse_cols = ['inner_temp', 'inner_hum']
    
    # 시간 특성 (추가 입력으로 사용)
    time_cols = [col for col in df.columns if col in ['hour', 'day', 'month', 'dayofweek', 
                 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                 'day_sin', 'day_cos', 'season', 'is_weekend']]
    
    # 기상 데이터에 시간 특성 추가
    weather_cols = weather_cols + time_cols
    
    # lag 특성도 온실 입력에 추가
    lag_cols = [col for col in df.columns if 'lag' in col]
    greenhouse_cols = greenhouse_cols + lag_cols
    
    print(f"\n입력 특성:")
    print(f"  - 기상 데이터: {len(weather_cols)}개")
    print(f"    예시: {weather_cols[:5]}")
    print(f"  - 온실 데이터: {len(greenhouse_cols)}개")
    print(f"    예시: {greenhouse_cols}")
    
    # 타겟 변수 (온실 온도, 습도)
    target_cols = ['inner_temp', 'inner_hum']
    print(f"\n출력 타겟: {target_cols}")
    
    # 시퀀스 생성
    sequence_length = 6  # 과거 6시간의 기상 데이터
    X_weather_list = []
    X_greenhouse_list = []
    y_list = []
    
    for i in range(len(df) - sequence_length - prediction_hours + 1):
        # 입력: 현재까지 6시간의 기상 데이터
        weather_seq = df[weather_cols].iloc[i:i+sequence_length].values
        
        # 입력: 현재 시점의 온실 상태
        greenhouse_current = df[greenhouse_cols].iloc[i+sequence_length-1].values
        
        # 출력: N시간 후의 온실 온습도
        target_future = df[target_cols].iloc[i+sequence_length+prediction_hours-1].values
        
        X_weather_list.append(weather_seq)
        X_greenhouse_list.append(greenhouse_current)
        y_list.append(target_future)
    
    X_weather = np.array(X_weather_list)
    X_greenhouse = np.array(X_greenhouse_list)
    y = np.array(y_list)
    
    print(f"\n생성된 데이터:")
    print(f"  X_weather: {X_weather.shape} (샘플, 시퀀스, 기상특성)")
    print(f"  X_greenhouse: {X_greenhouse.shape} (샘플, 온실특성)")
    print(f"  y: {y.shape} (샘플, 타겟)")
    print(f"  총 샘플: {len(y):,}개")
    
    return X_weather, X_greenhouse, y, weather_cols, greenhouse_cols, target_cols

# ===== 학습 함수 =====
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_weather, X_greenhouse, y in dataloader:
        X_weather = X_weather.to(device)
        X_greenhouse = X_greenhouse.to(device)
        y = y.to(device)
        
        # Forward
        outputs = model(X_weather, X_greenhouse)
        loss = criterion(outputs, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ===== 검증 함수 =====
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_weather, X_greenhouse, y in dataloader:
            X_weather = X_weather.to(device)
            X_greenhouse = X_greenhouse.to(device)
            y = y.to(device)
            
            outputs = model(X_weather, X_greenhouse)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    return total_loss / len(dataloader), predictions, actuals

# ===== 메인 학습 프로세스 =====
def main():
    print("=" * 70)
    print("온실 온습도 예측 LSTM 모델 학습")
    print("=" * 70)
    print(f"학습 데이터: 4/16-10/27 기상청 ASOS + 온실 데이터")
    print(f"예측 목표: {PREDICTION_HOURS}시간 후 온실 온도/습도")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n[1/8] 데이터 로딩...")
    df = pd.read_csv('output/preprocessed_data.csv')
    
    if 'Date&Time' in df.columns:
        print(f"데이터 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
        df = df.drop(columns=['Date&Time'])
    
    print(f"데이터 shape: {df.shape}")
    print(f"총 레코드: {len(df):,}개")
    
    # 2. 데이터 준비
    print("\n[2/8] 데이터 준비...")
    X_weather, X_greenhouse, y, weather_cols, greenhouse_cols, target_cols = \
        prepare_data(df, PREDICTION_HOURS)
    
    # 3. 스케일링
    print("\n[3/8] 데이터 스케일링...")
    
    # 기상 데이터 스케일링
    scaler_weather = StandardScaler()
    X_weather_shape = X_weather.shape
    X_weather_2d = X_weather.reshape(-1, X_weather.shape[-1])
    X_weather_scaled = scaler_weather.fit_transform(X_weather_2d)
    X_weather_scaled = X_weather_scaled.reshape(X_weather_shape)
    
    # 온실 현재 상태 스케일링
    scaler_greenhouse = StandardScaler()
    X_greenhouse_scaled = scaler_greenhouse.fit_transform(X_greenhouse)
    
    # 타겟 스케일링
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    print("✓ 스케일링 완료")
    
    # 4. 데이터 분할
    print("\n[4/8] 학습/검증 데이터 분할 (80:20)...")
    
    # 시계열이므로 shuffle=False
    split_idx = int(len(y) * 0.8)
    
    X_weather_train = X_weather_scaled[:split_idx]
    X_weather_val = X_weather_scaled[split_idx:]
    
    X_greenhouse_train = X_greenhouse_scaled[:split_idx]
    X_greenhouse_val = X_greenhouse_scaled[split_idx:]
    
    y_train = y_scaled[:split_idx]
    y_val = y_scaled[split_idx:]
    
    print(f"Train 샘플: {len(y_train):,}개")
    print(f"Val 샘플: {len(y_val):,}개")
    
    # 데이터로더 생성
    train_dataset = GreenhouseDataset(X_weather_train, X_greenhouse_train, y_train)
    val_dataset = GreenhouseDataset(X_weather_val, X_greenhouse_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. 모델 초기화
    print("\n[5/8] 모델 초기화...")
    
    weather_input_size = X_weather.shape[2]
    greenhouse_input_size = X_greenhouse.shape[1]
    output_size = len(target_cols)
    
    model = GreenhousePredictionLSTM(
        weather_input_size=weather_input_size,
        greenhouse_input_size=greenhouse_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\n모델 구조:")
    print(f"  - 기상 입력 크기: {weather_input_size}")
    print(f"  - 온실 입력 크기: {greenhouse_input_size}")
    print(f"  - Hidden size: {HIDDEN_SIZE}")
    print(f"  - LSTM layers: {NUM_LAYERS}")
    print(f"  - 출력 크기: {output_size}")
    print(f"  - 총 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    # 6. 손실함수 및 옵티마이저
    print("\n[6/8] 옵티마이저 설정...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 7. 학습
    print(f"\n[7/8] 모델 학습 시작 ({EPOCHS} epochs)...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 25
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_actuals = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 로그 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 실제 스케일로 변환하여 MAE 계산
            val_preds_real = scaler_y.inverse_transform(val_preds)
            val_actuals_real = scaler_y.inverse_transform(val_actuals)
            
            mae_temp = np.mean(np.abs(val_preds_real[:, 0] - val_actuals_real[:, 0]))
            mae_hum = np.mean(np.abs(val_preds_real[:, 1] - val_actuals_real[:, 1]))
            
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"Loss: Train {train_loss:.6f} | Val {val_loss:.6f} | "
                  f"MAE: 온도 {mae_temp:.2f}°C, 습도 {mae_hum:.2f}%")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'weather_input_size': weather_input_size,
                    'greenhouse_input_size': greenhouse_input_size,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'output_size': output_size,
                    'dropout': DROPOUT,
                    'prediction_hours': PREDICTION_HOURS,
                    'weather_cols': weather_cols,
                    'greenhouse_cols': greenhouse_cols,
                    'target_cols': target_cols
                }
            }, 'output/best_model.pth')
            
            if (epoch + 1) % 5 == 0:
                print(f"  ✓ Best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    # 8. 저장
    print("\n[8/8] 스케일러 및 메타데이터 저장...")
    os.makedirs('output/cache', exist_ok=True)
    
    # 스케일러 저장
    with open('output/cache/scaler_weather.pkl', 'wb') as f:
        pickle.dump(scaler_weather, f)
    with open('output/cache/scaler_greenhouse.pkl', 'wb') as f:
        pickle.dump(scaler_greenhouse, f)
    with open('output/cache/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # 메타데이터 저장
    metadata = {
        'prediction_hours': PREDICTION_HOURS,
        'weather_cols': weather_cols,
        'greenhouse_cols': greenhouse_cols,
        'target_cols': target_cols,
        'sequence_length': 6
    }
    with open('output/cache/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # 학습 이력 저장
    results = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    results.to_csv('output/training_history.csv', index=False)
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    print(f"최고 검증 손실: {best_val_loss:.6f}")
    print(f"예측 시간: {PREDICTION_HOURS}시간 후")
    print(f"\n저장된 파일:")
    print(f"  📦 모델: output/best_model.pth")
    print(f"  📊 스케일러: output/cache/scaler_*.pkl")
    print(f"  📋 메타데이터: output/cache/metadata.pkl")
    print(f"  📈 학습 이력: output/training_history.csv")
    print("=" * 70)
    print("\n💡 다음 단계: future_prediction.py로 단기예보 데이터 예측")

if __name__ == "__main__":
    main()