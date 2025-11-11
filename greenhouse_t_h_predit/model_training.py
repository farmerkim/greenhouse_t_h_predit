# model_training.py
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM 모델"""
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
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # 마지막 시퀀스만 사용
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class LSTMGreenhousePredictor:
    def __init__(self, time_steps=24):
        self.time_steps = time_steps
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.feature_columns = None
        self.target_columns = ['inner_temp', 'inner_hum']
        self.device = device
        
    def prepare_data(self, data_path, train_end_date, test_start_date, test_end_date):
        """데이터 로드 및 분할"""
        print("="*60)
        print("데이터 준비 중...")
        print("="*60)
        
        # 전처리된 데이터 로드
        data = pd.read_csv(data_path, parse_dates=['Date&Time'])
        
        print(f"총 데이터: {len(data)} 레코드")
        print(f"기간: {data['Date&Time'].min()} ~ {data['Date&Time'].max()}")
        print(f"컬럼: {list(data.columns)}")
        
        # 결측치 확인
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print(f"\n결측치 발견:")
            print(missing[missing > 0])
            data = data.dropna()
            print(f"결측치 제거 후: {len(data)} 레코드")
        
        # 학습/테스트 분할
        train_data = data[data['Date&Time'] <= train_end_date].copy()
        test_data = data[(data['Date&Time'] >= test_start_date) & 
                        (data['Date&Time'] <= test_end_date)].copy()
        
        print(f"\n학습 데이터: {len(train_data)} 레코드")
        print(f"  기간: {train_data['Date&Time'].min()} ~ {train_data['Date&Time'].max()}")
        print(f"테스트 데이터: {len(test_data)} 레코드")
        print(f"  기간: {test_data['Date&Time'].min()} ~ {test_data['Date&Time'].max()}")
        
        if len(train_data) < self.time_steps:
            raise ValueError(f"학습 데이터가 부족합니다. 최소 {self.time_steps}개 필요")
        
        if len(test_data) < self.time_steps:
            raise ValueError(f"테스트 데이터가 부족합니다. 최소 {self.time_steps}개 필요")
        
        # 특성 컬럼 선택
        exclude_cols = ['Date&Time']
        self.feature_columns = [col for col in data.columns 
                               if col not in exclude_cols and 
                               data[col].dtype in [np.float64, np.int64]]
        
        print(f"\n특성 수: {len(self.feature_columns)}")
        print(f"특성 목록: {self.feature_columns}")
        print(f"예측 대상: {self.target_columns}")
        
        # 데이터 정규화
        X_train = self.scaler_X.fit_transform(train_data[self.feature_columns])
        X_test = self.scaler_X.transform(test_data[self.feature_columns])
        
        y_train = self.scaler_y.fit_transform(train_data[self.target_columns])
        y_test = self.scaler_y.transform(test_data[self.target_columns])
        
        # 시퀀스 생성
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        print(f"\n시퀀스 생성 완료:")
        print(f"  학습 시퀀스: {X_train_seq.shape}")
        print(f"  테스트 시퀀스: {X_test_seq.shape}")
        
        # 타임스탬프 저장
        train_timestamps = train_data['Date&Time'].values[self.time_steps:]
        test_timestamps = test_data['Date&Time'].values[self.time_steps:]
        
        return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
                train_timestamps, test_timestamps, train_data, test_data)
    
    def create_sequences(self, X, y):
        """시계열 시퀀스 생성"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.time_steps):
            X_seq.append(X[i:i + self.time_steps])
            y_seq.append(y[i + self.time_steps])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
        """모델 학습"""
        print("\n" + "="*60)
        print("LSTM 모델 학습 시작")
        print("="*60)
        
        # models 디렉토리 생성
        os.makedirs('models', exist_ok=True)
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        input_size = X_train.shape[2]
        self.model = LSTMModel(input_size=input_size, output_size=len(self.target_columns))
        self.model = self.model.to(self.device)
        
        print(f"\n모델 구조:")
        print(self.model)
        print(f"\n파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False  # verbose=False로 변경
        )
        
        # 학습
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0
            train_mae = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += nn.L1Loss()(outputs, y_batch).item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # 검증 모드
            self.model.eval()
            val_loss = 0
            val_mae = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    val_mae += nn.L1Loss()(outputs, y_batch).item()
            
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            
            # 학습률 조정
            scheduler.step(val_loss)
            
            # 히스토리 저장
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # 로그 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly Stopping at epoch {epoch+1}")
                    break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        print(f"\n최고 성능 모델 로드 완료 (Val Loss: {best_val_loss:.4f})")
        
        return history
    
    def predict(self, X):
        """예측"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def evaluate(self, X_test, y_test_scaled):
        """모델 평가"""
        y_pred_scaled = self.predict(X_test)
        
        # 스케일 복원
        y_test = self.scaler_y.inverse_transform(y_test_scaled)
        y_pred = y_pred_scaled
        
        # 메트릭 계산
        metrics = {}
        for i, target in enumerate(self.target_columns):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            metrics[target] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\n{target} 예측 성능:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R² Score: {r2:.4f}")
        
        return metrics, y_pred
    
    def plot_training_history(self, history, save_path='models/training_history.png'):
        """학습 과정 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(history['train_mae'], label='Train MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n학습 그래프 저장: {save_path}")
        plt.close()
    
    def save_model(self, model_dir='models'):
        """모델 저장"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 저장
        torch.save(self.model.state_dict(), f'{model_dir}/lstm_model.pth')
        
        # 스케일러 저장
        with open(f'{model_dir}/scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        with open(f'{model_dir}/scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        # 설정 저장
        config = {
            'time_steps': self.time_steps,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        
        with open(f'{model_dir}/config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"\n✅ 모델 저장 완료: {model_dir}/")


def main():
    # 파라미터 설정
    DATA_PATH = 'output/preprocessed_data.csv'
    
    TIME_STEPS = 24
    # 4월 16일 ~ 10월 26일: 학습 및 테스트 데이터
    # 10월 27일: 예측 목표 (미래 1일)
    TRAIN_END_DATE = '2025-09-25 23:00:00'
    TEST_START_DATE = '2025-09-26 00:00:00'
    TEST_END_DATE = '2025-10-26 23:00:00'
    
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    print("="*60)
    print("모델 학습 및 성능 검증 모드 (80:20 분할)")
    print("="*60)
    print(f"학습/검증 기간: 4월 16일 ~ 9월 25일")
    print(f"최종 테스트 기간: 9월 26일 ~ 10월 26일")
    print("="*60)
    
    # 모델 초기화
    predictor = LSTMGreenhousePredictor(time_steps=TIME_STEPS)
    
    # 데이터 준비
    (X_train, y_train, X_test, y_test, 
     train_timestamps, test_timestamps, 
     train_data, test_data) = predictor.prepare_data(
        DATA_PATH, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
    )
    
    # 검증 데이터 분할
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"\n최종 데이터 크기:")
    print(f"  학습: {len(X_train)}")
    print(f"  검증: {len(X_val)}")
    print(f"  테스트: {len(X_test)}")
    
    # 모델 학습
    history = predictor.train(
        X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # 학습 과정 시각화
    predictor.plot_training_history(history)
    
    # 모델 평가
    print("\n" + "="*60)
    print("모델 평가")
    print("="*60)
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    
    # 모델 저장
    predictor.save_model()
    
    # 예측 결과 저장
    y_test_original = predictor.scaler_y.inverse_transform(y_test)
    
    predictions = {
        'y_test': y_test_original,
        'y_pred': y_pred,
        'test_timestamps': test_timestamps,
        'metrics': metrics,
        'target_columns': predictor.target_columns
    }
    
    with open('models/predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    
    print("\n✅ 예측 결과 저장 완료: models/predictions.pkl")
    
    # 결과 요약
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)
    print(f"프레임워크: PyTorch")
    print(f"디바이스: {device}")
    print(f"Time Steps: {TIME_STEPS}")
    print(f"특성 수: {len(predictor.feature_columns)}")
    print(f"\n성능 지표:")
    for target, metric in metrics.items():
        print(f"\n{target}:")
        print(f"  MAE: {metric['mae']:.4f}")
        print(f"  RMSE: {metric['rmse']:.4f}")
        print(f"  R²: {metric['r2']:.4f}")


if __name__ == "__main__":
    main()