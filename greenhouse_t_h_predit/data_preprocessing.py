import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_and_merge_data(greenhouse_path, asos_weather_path):
    """
    온실 데이터와 과거 ASOS 기상 데이터 로드 및 병합
    
    Args:
        greenhouse_path (str): 온실 센서 데이터 경로
        asos_weather_path (str): 과거 ASOS 기상 데이터 경로
    Returns:
        pd.DataFrame: 병합된 데이터
    """
    print("="*60)
    print("데이터 로드 및 병합")
    print("="*60)
    
    # 1. 온실 데이터 로드
    print("\n[1] 온실 센서 데이터 로드")
    if not os.path.exists(greenhouse_path):
        raise FileNotFoundError(f"온실 데이터 파일을 찾을 수 없습니다: {greenhouse_path}")
    
    greenhouse_df = pd.read_csv(greenhouse_path)
    
    # 컬럼 목록 출력
    print(f"  원본 컬럼: {list(greenhouse_df.columns)}")
    
    # Date&Time 컬럼 찾기
    time_col = None
    possible_time_cols = [
        'DATE&TIME', 'Date&Time', 'date&time', 'datetime', 
        'DATE', 'Date', 'date', 'TIME', 'Time', 'time',
        'timestamp', 'Datetime', 'dt', '일시', '날짜', '시간'
    ]
    
    # 정확히 일치하는 컬럼 찾기
    for col in possible_time_cols:
        if col in greenhouse_df.columns:
            time_col = col
            break
    
    # 부분 일치 시도 (대소문자 무시)
    if time_col is None:
        for col in greenhouse_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'datetime', '일시', '날짜']):
                time_col = col
                break
    
    # 첫 번째 컬럼 자동 감지
    if time_col is None:
        first_col = greenhouse_df.columns[0]
        try:
            # 날짜 형식 시도
            pd.to_datetime(greenhouse_df[first_col].iloc[0])
            time_col = first_col
            print(f"  🔍 첫 번째 컬럼 '{first_col}'이(가) 시간 데이터로 감지됨")
        except:
            raise ValueError(f"시간 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(greenhouse_df.columns)}")
    
    print(f"  ✓ 시간 컬럼 발견: '{time_col}'")
    
    # datetime으로 변환 (여러 형식 지원)
    try:
        # MM/DD/YYYY HH:MM 형식 (예: 04/15/2025 17:00)
        greenhouse_df['Date&Time'] = pd.to_datetime(greenhouse_df[time_col], format='%m/%d/%Y %H:%M')
    except:
        try:
            # 자동 감지
            greenhouse_df['Date&Time'] = pd.to_datetime(greenhouse_df[time_col])
        except Exception as e:
            print(f"  ❌ 시간 변환 실패: {e}")
            print(f"  첫 5개 값: {greenhouse_df[time_col].head()}")
            raise
    
    # 원래 시간 컬럼 제거
    if time_col != 'Date&Time':
        greenhouse_df = greenhouse_df.drop(time_col, axis=1)
    
    # 컬럼명 표준화 (Air_Temp_Avg → inner_temp, RH_Avg → inner_hum)
    column_mapping = {
        'Air_Temp_Avg': 'inner_temp',
        'air_temp_avg': 'inner_temp',
        'AirTemp': 'inner_temp',
        'Temp': 'inner_temp',
        'Temperature': 'inner_temp',
        '온도': 'inner_temp',
        '내부온도': 'inner_temp',
        
        'RH_Avg': 'inner_hum',
        'rh_avg': 'inner_hum',
        'RH': 'inner_hum',
        'Humidity': 'inner_hum',
        '습도': 'inner_hum',
        '내부습도': 'inner_hum'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in greenhouse_df.columns:
            greenhouse_df = greenhouse_df.rename(columns={old_name: new_name})
    
    print(f"  ✓ 온실 데이터: {greenhouse_df.shape}")
    print(f"  ✓ 컬럼: {list(greenhouse_df.columns)}")
    print(f"  ✓ 기간: {greenhouse_df['Date&Time'].min()} ~ {greenhouse_df['Date&Time'].max()}")
    
    # inner_temp와 inner_hum이 있는지 확인
    if 'inner_temp' not in greenhouse_df.columns or 'inner_hum' not in greenhouse_df.columns:
        print(f"\n  ⚠️  필수 컬럼이 없습니다.")
        print(f"  현재 컬럼: {list(greenhouse_df.columns)}")
        print(f"  필요: inner_temp (온실 온도), inner_hum (온실 습도)")
    
    # 2. ASOS 기상 데이터 로드
    print("\n[2] ASOS 기상 데이터 로드 (학습용)")
    if not os.path.exists(asos_weather_path):
        raise FileNotFoundError(f"⚠️  ASOS 기상 데이터 파일을 찾을 수 없습니다: {asos_weather_path}\n"
                              f"    학습을 위해서는 과거 기상 데이터가 필요합니다!")
    
    asos_df = pd.read_csv(asos_weather_path)
    
    print(f"  원본 컬럼: {list(asos_df.columns)}")
    
    # ASOS 데이터의 시간 컬럼 찾기
    asos_time_col = None
    for col in ['Date&Time', 'date', 'datetime', 'tm', 'time', 'timestamp']:
        if col in asos_df.columns:
            asos_time_col = col
            break
    
    if asos_time_col is None:
        for col in asos_df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                asos_time_col = col
                break
    
    if asos_time_col is None:
        raise ValueError("ASOS 데이터에서 시간 컬럼을 찾을 수 없습니다.")
    
    print(f"  ✓ 시간 컬럼 발견: '{asos_time_col}'")
    
    asos_df['Date&Time'] = pd.to_datetime(asos_df[asos_time_col])
    if asos_time_col != 'Date&Time':
        asos_df = asos_df.drop(asos_time_col, axis=1)
    
    print(f"  ✓ ASOS 데이터: {asos_df.shape}")
    print(f"  ✓ 기간: {asos_df['Date&Time'].min()} ~ {asos_df['Date&Time'].max()}")
    
    # ASOS 데이터 컬럼 표준화
    asos_column_mapping = {
        'ta': 'outer_temp',
        'temp': 'outer_temp',
        'temperature': 'outer_temp',
        'hm': 'outer_hum',
        'rh': 'outer_hum',
        'humidity': 'outer_hum',
        'ws': 'wind_speed',
        'wind': 'wind_speed',
        'wd': 'wind_dir',
        'wind_direction': 'wind_dir',
        'rn': 'rainfall',
        'precipitation': 'rainfall',
        'rain': 'rainfall',
        'si': 'solar_rad',
        'solar': 'solar_rad',
        'radiation': 'solar_rad',
        'icsr': 'solar_rad',
        'ps': 'pressure',
    }
    
    for old_name, new_name in asos_column_mapping.items():
        if old_name in asos_df.columns:
            asos_df = asos_df.rename(columns={old_name: new_name})
    
    print(f"  ✓ ASOS 컬럼 표준화 완료")
    
    # 3. 데이터 병합
    print("\n[3] 온실 데이터 + ASOS 기상 데이터 병합")
    
    # 병합 전 데이터 정렬
    greenhouse_df = greenhouse_df.sort_values('Date&Time')
    asos_df = asos_df.sort_values('Date&Time')
    
    # 병합 (merge_asof 사용)
    merged_df = pd.merge_asof(
        greenhouse_df,
        asos_df,
        on='Date&Time',
        direction='nearest',
        tolerance=pd.Timedelta('1H')
    )
    
    # 기상 데이터 결측치 처리
    weather_cols = ['outer_temp', 'outer_hum', 'wind_speed', 'wind_dir', 
                   'rainfall', 'solar_rad', 'pressure']
    
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
            if merged_df[col].isnull().sum() > 0:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    print(f"  ✓ 병합 완료: {merged_df.shape}")
    
    merged_weather_cols = [col for col in weather_cols if col in merged_df.columns]
    print(f"  ✓ 사용 가능한 기상 특성: {merged_weather_cols}")
    
    return merged_df


def add_time_features(df):
    """시간 관련 특성 추가"""
    print("\n[4] 시간 특성 추가")
    
    df = df.copy()
    
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
    
    df['season'] = df['month'].apply(lambda x: 
        0 if x in [12, 1, 2] else
        1 if x in [3, 4, 5] else
        2 if x in [6, 7, 8] else 3
    )
    
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    print(f"  ✓ 시간 특성 추가 완료")
    
    return df


def handle_missing_values(df):
    """결측치 처리"""
    print("\n[5] 결측치 처리")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  ⚠️  결측치 발견:")
        for col, count in missing[missing > 0].items():
            print(f"    - {col}: {count}개 ({count/len(df)*100:.2f}%)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            if df[col].isnull().sum() > 0:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(mean_val)
        
        print(f"  ✓ 결측치 처리 완료")
    else:
        print(f"  ✓ 결측치 없음")
    
    return df


def handle_outliers(df, columns=None, method='iqr', threshold=3):
    """이상치 처리"""
    print("\n[6] 이상치 처리")
    
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['hour', 'day', 'month', 'dayofweek', 'season', 'is_weekend',
                       'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        columns = [col for col in columns if col not in exclude_cols]
    
    outlier_count = 0
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outliers = z_scores > threshold
        
        if outliers.sum() > 0:
            outlier_count += outliers.sum()
            median_value = df[col].median()
            df.loc[outliers, col] = median_value
    
    print(f"  ✓ 이상치 처리 완료: {outlier_count}개 값 수정")
    
    return df


def create_lag_features(df, target_cols=['inner_temp', 'inner_hum'], lags=[1, 3, 6]):
    """지연(lag) 특성 생성"""
    print("\n[7] 지연(Lag) 특성 생성")
    
    df = df.copy()
    lag_cols_created = []
    
    for col in target_cols:
        if col not in df.columns:
            print(f"  ⚠️  컬럼을 찾을 수 없음: {col}")
            continue
        
        for lag in lags:
            lag_col_name = f'{col}_lag_{lag}'
            df[lag_col_name] = df[col].shift(lag)
            lag_cols_created.append(lag_col_name)
    
    print(f"  ✓ 생성된 lag 특성: {len(lag_cols_created)}개")
    
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"  ✓ Lag 결측치 제거: {removed}개 행")
    
    return df


def preprocess_data(greenhouse_path, asos_weather_path, output_path, 
                    add_lags=True, lag_hours=[1, 3, 6],
                    handle_outlier=True, outlier_method='iqr'):
    """전체 전처리 파이프라인"""
    print("\n" + "="*60)
    print("전처리 파이프라인 시작")
    print("="*60)
    
    df = load_and_merge_data(greenhouse_path, asos_weather_path)
    df = add_time_features(df)
    df = handle_missing_values(df)
    
    if handle_outlier:
        df = handle_outliers(df, method=outlier_method, threshold=3)
    
    if add_lags:
        df = create_lag_features(df, target_cols=['inner_temp', 'inner_hum'], lags=lag_hours)
    
    df = df.sort_values('Date&Time').reset_index(drop=True)
    
    print("\n[8] 전처리 완료")
    print(f"  ✓ 최종 데이터: {df.shape}")
    print(f"  ✓ 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
    print(f"  ✓ 컬럼: {list(df.columns)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    print(f"\n📊 주요 통계:")
    stats_cols = [col for col in ['inner_temp', 'inner_hum', 'outer_temp', 'outer_hum'] 
                  if col in df.columns]
    if stats_cols:
        print(df[stats_cols].describe())
    
    return df


def main():
    """메인 함수"""
    
    print("="*60)
    print("🌱 온실 미기후 데이터 전처리")
    print("="*60)
    
    GREENHOUSE_PATH = 'input/greenhouse_inner.csv'
    ASOS_WEATHER_PATH = 'input/asos_weather.csv'
    OUTPUT_PATH = 'output/preprocessed_data.csv'
    
    if not os.path.exists(GREENHOUSE_PATH):
        print(f"\n❌ 파일 없음: {GREENHOUSE_PATH}")
        return None
    
    if not os.path.exists(ASOS_WEATHER_PATH):
        print(f"\n❌ 파일 없음: {ASOS_WEATHER_PATH}")
        print("먼저 'python asos_download.py'를 실행하세요.")
        return None
    
    try:
        df = preprocess_data(
            greenhouse_path=GREENHOUSE_PATH,
            asos_weather_path=ASOS_WEATHER_PATH,
            output_path=OUTPUT_PATH,
            add_lags=True,
            lag_hours=[1, 3, 6],
            handle_outlier=True,
            outlier_method='iqr'
        )
        
        print("\n" + "="*60)
        print("✅ 전처리 완료!")
        print("="*60)
        print("\n💡 다음 단계: python model_training.py")
        
        return df
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()