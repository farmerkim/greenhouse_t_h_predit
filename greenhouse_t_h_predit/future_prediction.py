import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import os

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GreenhousePredictionLSTM(nn.Module):
    """
    학습된 모델과 동일한 구조
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


class GreenhouseFuturePredictor:
    """온실 미기후 미래 예측기"""
    
    def __init__(self, model_path='output/best_model.pth', 
                 scaler_dir='output/cache'):
        self.model_path = model_path
        self.scaler_dir = scaler_dir
        self.model = None
        self.scaler_weather = None
        self.scaler_greenhouse = None
        self.scaler_y = None
        self.metadata = None
        self.device = device
    
    def load_model_and_scalers(self):
        """저장된 모델, 스케일러, 메타데이터 로드"""
        print("="*60)
        print("모델 및 스케일러 로드")
        print("="*60)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        # 모델 초기화
        self.model = GreenhousePredictionLSTM(
            weather_input_size=config['weather_input_size'],
            greenhouse_input_size=config['greenhouse_input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 스케일러 로드
        with open(f'{self.scaler_dir}/scaler_weather.pkl', 'rb') as f:
            self.scaler_weather = pickle.load(f)
        with open(f'{self.scaler_dir}/scaler_greenhouse.pkl', 'rb') as f:
            self.scaler_greenhouse = pickle.load(f)
        with open(f'{self.scaler_dir}/scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)
        
        # 메타데이터 로드
        with open(f'{self.scaler_dir}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        return True
    
    def add_time_features(self, df):
        """시간 특성 추가"""
        df = df.copy()
        if df['Date&Time'].dtype == 'object':
            df['Date&Time'] = pd.to_datetime(df['Date&Time'])
        
        df['hour'] = df['Date&Time'].dt.hour
        df['day'] = df['Date&Time'].dt.day
        df['month'] = df['Date&Time'].dt.month
        df['dayofweek'] = df['Date&Time'].dt.dayofweek
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        df['season'] = df['month'].apply(lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        return df
    
    def get_recent_greenhouse_data(self, preprocessed_path):
        """최근 온실 데이터 가져오기"""
        df = pd.read_csv(preprocessed_path)
        last_row = df.iloc[-1]
        
        greenhouse_data = {}
        for col in self.metadata['greenhouse_cols']:
            if col in last_row:
                greenhouse_data[col] = last_row[col]
            else:
                greenhouse_data[col] = 0.0
        return greenhouse_data
    
    def load_forecast_data(self, forecast_path):
        """기상청 예보 데이터 로드"""
        df = pd.read_csv(forecast_path)
        if 'Date&Time' in df.columns:
            df['Date&Time'] = pd.to_datetime(df['Date&Time'])
        return df.sort_values('Date&Time').reset_index(drop=True)
    
    def predict(self, forecast_path, preprocessed_path, target_date=None, hours_to_predict=6):
        """예측 수행"""
        print("="*60)
        print(f"온실 온습도 예측 시작")
        print("="*60)
        
        if self.model is None:
            self.load_model_and_scalers()
        
        sequence_length = self.metadata['sequence_length']
        weather_cols = self.metadata['weather_cols']
        greenhouse_cols = self.metadata['greenhouse_cols']
        
        # 1. 데이터 로드
        current_greenhouse = self.get_recent_greenhouse_data(preprocessed_path)
        forecast_df = self.load_forecast_data(forecast_path)
        forecast_df = self.add_time_features(forecast_df)
        
        # 2. 날짜 필터링 (target_date가 없으면 현재 시점 이후 데이터 모두 사용)
        if target_date:
            target_dt = pd.to_datetime(target_date)
            # 타겟 날짜의 데이터부터 시작 (이전 데이터는 시퀀스 구성을 위해 필요할 수 있지만, 여기선 간단히 해당일 이후로 필터)
            forecast_data = forecast_df[forecast_df['Date&Time'] >= target_dt].copy()
        else:
            forecast_data = forecast_df.copy()
            
        forecast_data = forecast_data.sort_values('Date&Time').reset_index(drop=True)
        
        # 3. [중요] 누락된 기상 컬럼 채우기 (solar_rad, pressure 등)
        for col in weather_cols:
            if col not in forecast_data.columns:
                print(f"⚠️ 예보 데이터에 '{col}' 컬럼 누락 -> 기본값 대체")
                if 'solar' in col or 'rad' in col:
                    forecast_data[col] = 0.0  # 일사량 0
                elif 'pressure' in col:
                    forecast_data[col] = 1013.0  # 표준기압
                else:
                    forecast_data[col] = 0.0
        
        # 4. 데이터 길이 확인
        if len(forecast_data) < sequence_length:
            print(f"❌ 데이터 부족: {len(forecast_data)}개 < 필요 {sequence_length}개")
            return None
            
        predictions = []
        
        # 5. 예측 루프
        # 예측 가능한 횟수: (전체 데이터 길이 - 시퀀스 길이 + 1) 과 (요청한 예측 시간) 중 작은 값
        max_predictions = min(hours_to_predict, len(forecast_data) - sequence_length + 1)
        
        print(f"🔄 총 {max_predictions}시간 예측 수행 중...")
        
        for i in range(max_predictions):
            # 입력 시퀀스 추출
            weather_seq = forecast_data[weather_cols].iloc[i : i + sequence_length].values
            
            # 현재 온실 상태 (고정값 사용)
            greenhouse_state = np.array([current_greenhouse[col] for col in greenhouse_cols])
            
            # 스케일링
            weather_seq_scaled = self.scaler_weather.transform(weather_seq)
            weather_seq_scaled = weather_seq_scaled.reshape(1, sequence_length, -1)
            
            greenhouse_state_scaled = self.scaler_greenhouse.transform(greenhouse_state.reshape(1, -1))
            
            # 텐서 변환
            weather_tensor = torch.FloatTensor(weather_seq_scaled).to(self.device)
            greenhouse_tensor = torch.FloatTensor(greenhouse_state_scaled).to(self.device)
            
            # 모델 예측
            with torch.no_grad():
                output_scaled = self.model(weather_tensor, greenhouse_tensor)
                output = self.scaler_y.inverse_transform(output_scaled.cpu().numpy())
            
            # 결과 저장
            prediction_time = forecast_data.iloc[i + sequence_length - 1]['Date&Time']
            
            pred_row = {
                'Date&Time': prediction_time,
                'Hours_Ahead': i + 1,
                'Predicted_inner_temp': output[0, 0],
                'Predicted_inner_hum': output[0, 1]
            }
            
            # 시각화를 위해 기상 정보도 함께 저장
            forecast_row = forecast_data.iloc[i + sequence_length - 1]
            for c in ['outer_temp', 'outer_hum', 'wind_speed', 'rainfall']:
                if c in forecast_row:
                    pred_row[c] = forecast_row[c]
                    
            predictions.append(pred_row)
            
        if not predictions:
            print("❌ 생성된 예측 결과가 없습니다.")
            return None
            
        results_df = pd.DataFrame(predictions)
        
        # 저장
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'output/prediction_result_{timestamp}.csv'
        results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 예측 완료 및 저장: {save_path}")
        return results_df

def main():
    # 테스트용 실행 코드
    MODEL_PATH = 'output/best_model.pth'
    SCALER_DIR = 'output/cache'
    FORECAST_PATH = 'input/weather_forecast.csv'
    PREPROCESSED_PATH = 'output/preprocessed_data.csv'
    
    predictor = GreenhouseFuturePredictor(MODEL_PATH, SCALER_DIR)
    
    if os.path.exists(FORECAST_PATH) and os.path.exists(PREPROCESSED_PATH):
        try:
            results = predictor.predict(FORECAST_PATH, PREPROCESSED_PATH)
            if results is not None:
                print(results)
        except Exception as e:
            print(f"오류: {e}")
    else:
        print("필요한 데이터 파일이 없습니다.")

if __name__ == '__main__':
    main()