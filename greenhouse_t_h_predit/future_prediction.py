import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import os

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        """
        초기화
        
        Args:
            model_path (str): 학습된 모델 경로
            scaler_dir (str): 스케일러 저장 디렉토리
        """
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
        
        # 모델 체크포인트 로드
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
        
        print(f"✅ 로드 완료")
        print(f"  - 시퀀스 길이: {self.metadata['sequence_length']}시간")
        print(f"  - 예측 간격: {self.metadata['prediction_hours']}시간")
        print(f"  - 기상 특성: {len(self.metadata['weather_cols'])}개")
        print(f"  - 온실 특성: {len(self.metadata['greenhouse_cols'])}개")
        print(f"  - 타겟: {self.metadata['target_cols']}")
        
        return True
    
    def add_time_features(self, df):
        """시간 특성 추가"""
        df = df.copy()
        
        # Date&Time이 문자열이면 datetime으로 변환
        if df['Date&Time'].dtype == 'object':
            df['Date&Time'] = pd.to_datetime(df['Date&Time'])
        
        # 기본 시간 특성
        df['hour'] = df['Date&Time'].dt.hour
        df['day'] = df['Date&Time'].dt.day
        df['month'] = df['Date&Time'].dt.month
        df['dayofweek'] = df['Date&Time'].dt.dayofweek
        
        # 순환 시간 특성
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 계절
        df['season'] = df['month'].apply(lambda x:
            0 if x in [12, 1, 2] else
            1 if x in [3, 4, 5] else
            2 if x in [6, 7, 8] else 3
        )
        
        # 주말 여부
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        return df
    
    def get_recent_greenhouse_data(self, preprocessed_path):
        """
        최근 온실 데이터 가져오기
        
        Args:
            preprocessed_path (str): 전처리된 데이터 경로
        Returns:
            dict: 최근 온실 상태
        """
        print("\n최근 온실 데이터 로드 중...")
        
        df = pd.read_csv(preprocessed_path)
        
        # 마지막 행 (가장 최근 데이터)
        last_row = df.iloc[-1]
        
        greenhouse_data = {}
        for col in self.metadata['greenhouse_cols']:
            if col in last_row:
                greenhouse_data[col] = last_row[col]
            else:
                print(f"  ⚠️  컬럼 누락: {col}, 기본값 사용")
                greenhouse_data[col] = 0.0
        
        print(f"  ✓ 로드 완료")
        print(f"  ✓ 현재 온실 온도: {greenhouse_data.get('inner_temp', 'N/A')}°C")
        print(f"  ✓ 현재 온실 습도: {greenhouse_data.get('inner_hum', 'N/A')}%")
        
        return greenhouse_data
    
    def load_forecast_data(self, forecast_path):
        """
        기상청 단기예보 데이터 로드
        
        Args:
            forecast_path (str): 예보 데이터 경로
        Returns:
            pd.DataFrame: 예보 데이터
        """
        print("\n기상청 단기예보 데이터 로드 중...")
        
        df = pd.read_csv(forecast_path)
        
        # Date&Time 컬럼 확인 및 변환
        if 'Date&Time' in df.columns:
            df['Date&Time'] = pd.to_datetime(df['Date&Time'])
        else:
            raise ValueError("예보 데이터에 'Date&Time' 컬럼이 없습니다.")
        
        df = df.sort_values('Date&Time').reset_index(drop=True)
        
        print(f"  ✓ 로드 완료: {len(df)}개 시간대")
        print(f"  ✓ 예보 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
        
        return df
    
    def predict(self, forecast_path, preprocessed_path, target_date=None, hours_to_predict=6):
        """
        특정 날짜의 기상예보를 사용하여 온실 온습도 예측
        
        Args:
            forecast_path (str): 기상청 예보 데이터 경로
            preprocessed_path (str): 전처리된 데이터 경로
            target_date (str): 예측할 날짜 (예: '2025-11-28'), None이면 가장 이른 날짜
            hours_to_predict (int): 예측할 시간 수 (기본 6시간)
        Returns:
            pd.DataFrame: 예측 결과
        """
        print("="*60)
        print(f"온실 온습도 예측 시작")
        print("="*60)
        
        # 1. 모델 로드
        if self.model is None:
            self.load_model_and_scalers()
        
        sequence_length = self.metadata['sequence_length']
        weather_cols = self.metadata['weather_cols']
        greenhouse_cols = self.metadata['greenhouse_cols']
        
        # 2. 현재 온실 상태 가져오기
        current_greenhouse = self.get_recent_greenhouse_data(preprocessed_path)
        
        # 3. 예보 데이터 로드
        forecast_df = self.load_forecast_data(forecast_path)
        
        # 4. 시간 특성 추가
        forecast_df = self.add_time_features(forecast_df)
        
        # 5. 대상 날짜 선택
        if target_date is None:
            # 가장 이른 날짜 사용
            target_date = forecast_df['Date&Time'].dt.date.min()
            print(f"\n대상 날짜: {target_date} (예보 데이터의 가장 이른 날짜)")
        else:
            target_date = pd.to_datetime(target_date).date()
            print(f"\n대상 날짜: {target_date}")
        
        # 해당 날짜의 데이터만 필터링
        forecast_day = forecast_df[forecast_df['Date&Time'].dt.date == target_date].copy()
        
        if len(forecast_day) == 0:
            print(f"❌ {target_date}에 해당하는 예보 데이터가 없습니다.")
            return None
        
        forecast_day = forecast_day.sort_values('Date&Time').reset_index(drop=True)
        print(f"  ✓ {target_date} 예보 데이터: {len(forecast_day)}개 시간대")
        
        # 6. 필요한 기상 컬럼 확인 및 생성
        for col in weather_cols:
            if col not in forecast_day.columns:
                if col.startswith('outer_') or col in ['wind_speed', 'wind_dir', 
                                                        'rainfall', 'solar_rad', 'pressure']:
                    print(f"  ⚠️  기상 컬럼 누락: {col}, 0으로 채움")
                    forecast_day[col] = 0.0
        
        # 7. 예측 수행
        print(f"\n{hours_to_predict}시간 예측 수행 중...")
        predictions = []
        
        # sequence_length 이상의 데이터가 필요
        if len(forecast_day) < sequence_length:
            print(f"❌ 데이터 부족: {len(forecast_day)}시간 < 필요 {sequence_length}시간")
            return None
        
        # 첫 시작점부터 sequence_length만큼 사용하여 예측
        for i in range(min(hours_to_predict, len(forecast_day) - sequence_length + 1)):
            # 기상 시퀀스: i 시점부터 sequence_length 만큼
            weather_seq = forecast_day[weather_cols].iloc[i:i+sequence_length].values
            
            # 온실 현재 상태
            greenhouse_state = np.array([current_greenhouse[col] 
                                        for col in greenhouse_cols])
            
            # 예측 시점: 시퀀스의 마지막 시점 기준
            prediction_time = forecast_day.iloc[i + sequence_length - 1]['Date&Time']
            
            # 스케일링
            weather_seq_scaled = self.scaler_weather.transform(
                weather_seq.reshape(-1, weather_seq.shape[-1])
            ).reshape(1, weather_seq.shape[0], weather_seq.shape[1])
            
            greenhouse_state_scaled = self.scaler_greenhouse.transform(
                greenhouse_state.reshape(1, -1)
            )
            
            # 텐서 변환
            weather_tensor = torch.FloatTensor(weather_seq_scaled).to(self.device)
            greenhouse_tensor = torch.FloatTensor(greenhouse_state_scaled).to(self.device)
            
            # 예측
            with torch.no_grad():
                output_scaled = self.model(weather_tensor, greenhouse_tensor)
                output = self.scaler_y.inverse_transform(
                    output_scaled.cpu().numpy()
                )
            
            # 결과 저장
            pred_data = {
                'Date&Time': prediction_time,
                'Hours_Ahead': i + 1,
                'Predicted_inner_temp': output[0, 0],
                'Predicted_inner_hum': output[0, 1]
            }
            
            # 해당 시점의 기상 정보 추가
            forecast_row = forecast_day.iloc[i + sequence_length - 1]
            if 'outer_temp' in forecast_row:
                pred_data['outer_temp'] = forecast_row['outer_temp']
            if 'outer_hum' in forecast_row:
                pred_data['outer_hum'] = forecast_row['outer_hum']
            if 'wind_speed' in forecast_row:
                pred_data['wind_speed'] = forecast_row['wind_speed']
            if 'rainfall' in forecast_row:
                pred_data['rainfall'] = forecast_row['rainfall']
            
            predictions.append(pred_data)
            
            print(f"  ✓ {i+1}시간 후 ({prediction_time}): "
                  f"온도 {output[0, 0]:.1f}°C, 습도 {output[0, 1]:.1f}%")
        
        # 8. 결과 정리
        if len(predictions) == 0:
            print("❌ 예측 결과가 없습니다.")
            return None
        
        results_df = pd.DataFrame(predictions)
        
        print("\n" + "="*60)
        print("✅ 예측 완료!")
        print("="*60)
        print(f"\n📊 예측 결과:")
        print(results_df.to_string(index=False))
        
        # 9. 저장
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'output/future_prediction_{target_date}_{timestamp}.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장 완료: {output_path}")
        
        # 10. 통계
        print(f"\n📈 예측 통계:")
        print(f"  온도 범위: {results_df['Predicted_inner_temp'].min():.1f}°C ~ "
              f"{results_df['Predicted_inner_temp'].max():.1f}°C")
        print(f"  습도 범위: {results_df['Predicted_inner_hum'].min():.1f}% ~ "
              f"{results_df['Predicted_inner_hum'].max():.1f}%")
        print(f"  평균 온도: {results_df['Predicted_inner_temp'].mean():.1f}°C")
        print(f"  평균 습도: {results_df['Predicted_inner_hum'].mean():.1f}%")
        
        return results_df


def main():
    """메인 실행 함수"""
    
    print("="*60)
    print("🌱 온실 미기후 미래 예측 시스템")
    print("="*60)
    
    # 설정
    MODEL_PATH = 'output/best_model.pth'
    SCALER_DIR = 'output/cache'
    FORECAST_PATH = 'input/weather_forecast.csv'  # 기상청 단기예보 데이터
    PREPROCESSED_PATH = 'output/preprocessed_data.csv'  # 전처리된 데이터
    TARGET_DATE = None  # None이면 예보의 가장 이른 날짜, 또는 '2025-11-28' 형식
    HOURS_TO_PREDICT = 6  # 예측할 시간 수
    
    # 필수 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 학습된 모델을 찾을 수 없습니다: {MODEL_PATH}")
        print("   model_training.py를 먼저 실행하세요.")
        return
    
    if not os.path.exists(FORECAST_PATH):
        print(f"❌ 예보 데이터를 찾을 수 없습니다: {FORECAST_PATH}")
        print("   기상청 예보 데이터를 준비하세요.")
        return
    
    if not os.path.exists(PREPROCESSED_PATH):
        print(f"❌ 전처리 데이터를 찾을 수 없습니다: {PREPROCESSED_PATH}")
        print("   data_preprocessing.py를 먼저 실행하세요.")
        return
    
    # 예측기 초기화
    predictor = GreenhouseFuturePredictor(
        model_path=MODEL_PATH,
        scaler_dir=SCALER_DIR
    )
    
    # 예측 실행
    try:
        results = predictor.predict(
            forecast_path=FORECAST_PATH,
            preprocessed_path=PREPROCESSED_PATH,
            target_date=TARGET_DATE,  # 예측할 날짜 지정
            hours_to_predict=HOURS_TO_PREDICT
        )
        
        if results is not None:
            print("\n🎉 예측이 성공적으로 완료되었습니다!")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()