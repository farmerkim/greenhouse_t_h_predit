import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

class WeatherForecastCollector:
    """기상청 단기예보 데이터 수집기"""
    
    def __init__(self, service_key):
        """
        기상청 단기예보 API 초기화
        
        Args:
            service_key (str): 공공데이터포털에서 발급받은 인증키
        """
        self.service_key = service_key
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        
        # 김제시 백구면 좌표
        self.nx = 60
        self.ny = 90
        
        # 카테고리 설명
        self.category_names = {
            'POP': 'rain_prob',          # 강수확률(%)
            'PTY': 'rain_type',          # 강수형태
            'PCP': 'rainfall',           # 1시간 강수량(mm)
            'REH': 'outer_hum',          # 습도(%)
            'SNO': 'snow',               # 1시간 신적설(cm)
            'SKY': 'sky_status',         # 하늘상태
            'TMP': 'outer_temp',         # 기온(℃)
            'TMN': 'min_temp',           # 최저기온(℃)
            'TMX': 'max_temp',           # 최고기온(℃)
            'UUU': 'wind_ew',            # 풍속-동서성분(m/s)
            'VVV': 'wind_ns',            # 풍속-남북성분(m/s)
            'WAV': 'wave_height',        # 파고(m)
            'VEC': 'wind_dir',           # 풍향(deg)
            'WSD': 'wind_speed',         # 풍속(m/s)
            'T1H': 'outer_temp',         # 기온(℃)
            'RN1': 'rainfall',           # 1시간 강수량(mm)
            'LGT': 'lightning'           # 낙뢰
        }
    
    def get_base_time_for_forecast(self, target_datetime):
        """
        단기예보의 발표시각 계산
        발표시각: 02, 05, 08, 11, 14, 17, 20, 23시 (하루 8회)
        API 제공: 발표시각 + 10분 이후
        
        Args:
            target_datetime: 목표 시간
        Returns:
            tuple: (base_date, base_time)
        """
        base_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
        
        hour = target_datetime.hour
        
        # 가장 최근 발표시각 찾기
        if hour < 2 or (hour == 2 and target_datetime.minute < 10):
            base_dt = target_datetime - timedelta(days=1)
            base_time = '2300'
        elif hour < 5 or (hour == 5 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0200'
        elif hour < 8 or (hour == 8 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0500'
        elif hour < 11 or (hour == 11 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0800'
        elif hour < 14 or (hour == 14 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1100'
        elif hour < 17 or (hour == 17 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1400'
        elif hour < 20 or (hour == 20 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1700'
        elif hour < 23 or (hour == 23 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '2000'
        else:
            base_dt = target_datetime
            base_time = '2300'
        
        base_date = base_dt.strftime('%Y%m%d')
        return base_date, base_time
    
    def get_vilage_fcst(self, target_date=None):
        """
        단기예보 조회 (3일 예보)
        
        Args:
            target_date: 조회할 날짜 (None이면 현재)
        Returns:
            dict: API 응답 데이터
        """
        url = f"{self.base_url}/getVilageFcst"
        
        if target_date is None:
            target_date = datetime.now()
        
        base_date, base_time = self.get_base_time_for_forecast(target_date)
        
        params = {
            'serviceKey': self.service_key,
            'numOfRows': 1000,
            'pageNo': 1,
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': self.nx,
            'ny': self.ny
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"단기예보 조회 오류: {e}")
            return None
    
    def parse_forecast_to_dataframe(self, forecast_data):
        """
        예보 데이터를 DataFrame으로 변환
        
        Args:
            forecast_data (dict): API 응답 데이터
        Returns:
            pd.DataFrame: 정리된 예보 데이터
        """
        if not forecast_data or 'response' not in forecast_data:
            return None
        
        response = forecast_data['response']
        
        # 에러 체크
        if response['header']['resultCode'] != '00':
            print(f"API 오류: {response['header']['resultMsg']}")
            return None
        
        items = response['body']['items']['item']
        
        # 시간별로 데이터 그룹화
        forecast_dict = {}
        
        for item in items:
            fcst_date = item.get('fcstDate', item.get('baseDate'))
            fcst_time = item.get('fcstTime', item.get('baseTime'))
            
            # 시간 키 생성
            time_key = f"{fcst_date}{fcst_time}"
            
            if time_key not in forecast_dict:
                forecast_dict[time_key] = {
                    'Date&Time': pd.to_datetime(time_key, format='%Y%m%d%H%M')
                }
            
            # 카테고리 매핑
            category = item['category']
            if category in self.category_names:
                col_name = self.category_names[category]
                value = item.get('fcstValue', item.get('obsrValue', '0'))
                
                # 숫자 변환
                try:
                    # 강수량 처리
                    if col_name == 'rainfall':
                        if '미만' in str(value):
                            value = 0.0
                        elif '이상' in str(value):
                            value = float(str(value).replace('mm이상', '').replace('이상', ''))
                        elif '~' in str(value):
                            # 범위값은 중간값 사용
                            parts = str(value).replace('mm', '').split('~')
                            value = (float(parts[0]) + float(parts[1])) / 2
                        else:
                            value = float(value)
                    else:
                        value = float(value)
                except:
                    value = 0.0
                
                forecast_dict[time_key][col_name] = value
        
        # DataFrame 생성
        df = pd.DataFrame.from_dict(forecast_dict, orient='index')
        df = df.reset_index(drop=True)
        df = df.sort_values('Date&Time').reset_index(drop=True)
        
        # 필요한 컬럼만 선택 (없는 컬럼은 0으로 채움)
        required_cols = ['outer_temp', 'outer_hum', 'wind_speed', 'rain_prob', 
                        'rainfall', 'sky_status', 'wind_dir']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # 컬럼 순서 정리
        cols = ['Date&Time'] + required_cols
        df = df[cols]
        
        return df
    
    def collect_historical_data(self, start_date, end_date, save_path='input/weather_data.csv'):
        """
        과거 기상 데이터 수집 (실제로는 API 제약으로 어려움)
        
        Args:
            start_date (str): 시작일 'YYYY-MM-DD'
            end_date (str): 종료일 'YYYY-MM-DD'
            save_path (str): 저장 경로
        """
        print("="*60)
        print("⚠️  주의: 기상청 단기예보 API는 과거 데이터를 제공하지 않습니다.")
        print("현재 시점의 예보 데이터만 수집 가능합니다.")
        print("="*60)
        
        # 현재 데이터만 수집
        current_data = self.get_vilage_fcst()
        df = self.parse_forecast_to_dataframe(current_data)
        
        if df is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"✅ 현재 기상 예보 데이터 저장: {save_path}")
            print(f"   수집된 데이터: {len(df)}개 시간대")
            print(f"   기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
        
        return df
    
    def get_current_forecast(self):
        """현재 기상 예보 조회 및 DataFrame 반환"""
        forecast_data = self.get_vilage_fcst()
        return self.parse_forecast_to_dataframe(forecast_data)


def create_weather_features_for_training(greenhouse_df, weather_df):
    """
    온실 데이터와 기상 데이터 병합 (학습용)
    
    Args:
        greenhouse_df (pd.DataFrame): 온실 센서 데이터
        weather_df (pd.DataFrame): 기상 예보 데이터
    Returns:
        pd.DataFrame: 병합된 데이터
    """
    print("\n기상 데이터 병합 중...")
    
    # 시간 인덱스 정렬
    greenhouse_df = greenhouse_df.copy()
    weather_df = weather_df.copy()
    
    # Date&Time 컬럼을 datetime으로 변환
    if 'Date&Time' in greenhouse_df.columns:
        greenhouse_df['Date&Time'] = pd.to_datetime(greenhouse_df['Date&Time'])
    
    if 'Date&Time' in weather_df.columns:
        weather_df['Date&Time'] = pd.to_datetime(weather_df['Date&Time'])
    
    # 병합 (left join으로 온실 데이터 기준)
    merged_df = pd.merge(
        greenhouse_df, 
        weather_df, 
        on='Date&Time', 
        how='left',
        suffixes=('', '_weather')
    )
    
    # 결측치 처리 (forward fill)
    weather_cols = ['outer_temp', 'outer_hum', 'wind_speed', 'rain_prob', 
                    'rainfall', 'sky_status', 'wind_dir']
    
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ 병합 완료: {len(merged_df)} 레코드")
    print(f"   기간: {merged_df['Date&Time'].min()} ~ {merged_df['Date&Time'].max()}")
    
    return merged_df


# 사용 예제
if __name__ == "__main__":
    # ⭐ 여기에 인증키를 입력하세요
    SERVICE_KEY = "c41d42c7c683c85b3e54a9bc00ec9d9e71f052d73a0722a759c14d40836f05cb"
    
    # WeatherForecastCollector 객체 생성
    weather_collector = WeatherForecastCollector(SERVICE_KEY)
    
    print("=" * 70)
    print("📍 지역: 전북특별자치도 김제시 백구면 (X:60, Y:90)")
    print("=" * 70)
    
    # 1. 현재 기상 예보 수집
    print("\n[1] 현재 기상 예보 데이터 수집")
    print("-" * 70)
    
    current_weather = weather_collector.get_current_forecast()
    
    if current_weather is not None:
        print(f"\n✅ 수집 완료:")
        print(f"   데이터 개수: {len(current_weather)}개 시간대")
        print(f"   기간: {current_weather['Date&Time'].min()} ~ {current_weather['Date&Time'].max()}")
        print(f"\n📊 첫 5개 데이터:")
        print(current_weather.head())
        
        # CSV 저장
        os.makedirs('input', exist_ok=True)
        save_path = 'input/weather_forecast.csv'
        current_weather.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장 완료: {save_path}")
        
        # 통계 정보
        print(f"\n📈 기상 데이터 통계:")
        print(current_weather.describe())
    
    print("\n" + "=" * 70)
    print("✅ 기상 데이터 수집 완료!")
    print("=" * 70)
    
    print("\n💡 다음 단계:")
    print("   1. 온실 센서 데이터를 input/greenhouse_inner.csv에 준비")
    print("   2. data_preprocessing.py로 데이터 전처리 실행")
    print("   3. model_training.py로 LSTM 모델 학습")