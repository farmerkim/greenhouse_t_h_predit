import pandas as pd
import numpy as np
import os

# ==============================================================================
# 🛠️ 수정된 load_and_merge_data 함수: ASOS 컬럼 중복 오류 최종 해결
# ==============================================================================
def load_and_merge_data(greenhouse_path, asos_weather_path):
    """
    온실 데이터와 과거 ASOS 기상 데이터 로드 및 병합
    """
    print("="*60)
    print("데이터 로드 및 병합")
    print("="*60)
    
    # 1. 온실 데이터 로드 (이전 수정 사항 유지)
    print("\n[1] 온실 센서 데이터 로드")
    if not os.path.exists(greenhouse_path):
        raise FileNotFoundError(f"온실 데이터 파일을 찾을 수 없습니다: {greenhouse_path}")
    
    delimiter = ';'
    print(f"  ✓ 구분자: 세미콜론(;)")
    
    try:
        new_column_names = [
            'Date&Time', 'release_cooling', 'meas_lee_vent_contr', 
            'radiation', 'radiation_sum', 'status_swi_cool', 
            'vent_temp', 'meas_curtain_2', 'inner_temp', 'inner_temp_2', 
            'set_heat_temp', 'set_vent_temp', 'outer_temp', 'heat_temp'
        ]
        
        greenhouse_df = pd.read_csv(
            greenhouse_path, 
            sep=delimiter, 
            skiprows=3, 
            names=new_column_names, 
            encoding='utf-8'
        )
        print(f"  ✓ CSV 로드: {greenhouse_df.shape} (skiprows=3, names 사용)")
        
    except Exception as e:
        print(f"  ❌ CSV 로드 실패: {e}")
        raise 
    
    if 'inner_temp_2' in greenhouse_df.columns:
        greenhouse_df = greenhouse_df.drop('inner_temp_2', axis=1)

    time_col = 'Date&Time'
    try:
        greenhouse_df[time_col] = pd.to_datetime(greenhouse_df[time_col], format='%d-%m-%Y %H:%M:%S')
        print("  ✓ 날짜 변환 성공 (DD-MM-YYYY HH:MM:SS)")
    except Exception:
        greenhouse_df[time_col] = pd.to_datetime(greenhouse_df[time_col], dayfirst=True)
        print("  ✓ 날짜 변환 성공 (자동 감지)")

    print(f"  ✓ 온실 온도: 'inner_temp' (사용)")
    
    if 'inner_hum' not in greenhouse_df.columns:
        print(f"  ⚠️  온실 습도 컬럼을 찾을 수 없습니다. 더미 데이터를 생성합니다.")
        greenhouse_df['inner_hum'] = 70.0
    
    required_cols = ['Date&Time', 'inner_temp', 'inner_hum', 'outer_temp', 
                     'radiation', 'radiation_sum', 'release_cooling', 
                     'meas_lee_vent_contr', 'status_swi_cool', 'vent_temp', 
                     'meas_curtain_2', 'set_heat_temp', 'set_vent_temp', 'heat_temp']
    
    available_cols = [col for col in required_cols if col in greenhouse_df.columns]
    greenhouse_df = greenhouse_df[available_cols]

    numeric_cols = [col for col in available_cols if col not in ['Date&Time']]
    for col in numeric_cols:
        greenhouse_df.loc[:, col] = pd.to_numeric(greenhouse_df[col], errors='coerce')

    print(f"\n  ✓ 온실 데이터: {greenhouse_df.shape}")
    
    # 5분 데이터를 1시간 데이터로 리샘플링
    print(f"\n  🔄 시간 단위로 리샘플링 (평균)...")
    greenhouse_df = greenhouse_df.set_index('Date&Time')
    greenhouse_df = greenhouse_df.resample('1H').mean().reset_index() 
    greenhouse_df = greenhouse_df.dropna(subset=['inner_temp', 'outer_temp'])
    print(f"  ✓ 리샘플링 후: {greenhouse_df.shape}")
    
    # ------------------------------------------------------------
    # 2. ASOS 기상 데이터 로드 (핵심 수정 부분)
    # ------------------------------------------------------------
    print("\n[2] ASOS 기상 데이터 로드 (학습용)")
    if not os.path.exists(asos_weather_path):
        raise FileNotFoundError(f"⚠️  ASOS 기상 데이터 파일을 찾을 수 없습니다: {asos_weather_path}")

    # **ASOS 데이터 로드 시 중복된 'Date&Time' 컬럼이 발생하지 않도록 컬럼 이름을 조정하며 로드**
    asos_df = pd.read_csv(asos_weather_path)
    print(f"  원본 컬럼: {list(asos_df.columns)}")
    
    # === 중복 Date&Time 컬럼 처리 로직 강화 ===
    
    # 1. 중복된 'Date&Time' 컬럼 인덱스 찾기
    col_names = asos_df.columns.tolist()
    duplicate_indices = [i for i, x in enumerate(col_names) if x == 'Date&Time']

    if len(duplicate_indices) > 1:
        print(f"  ⚠️  ASOS 데이터에 'Date&Time' 중복 컬럼 {len(duplicate_indices)}개 발견. 첫 번째 컬럼만 남깁니다.")
        # 첫 번째 컬럼만 True로, 나머지는 False인 boolean 마스크 생성
        keep_mask = [True] * len(col_names)
        for i in duplicate_indices[1:]:
            keep_mask[i] = False
        
        # 중복 컬럼을 제거한 새 DataFrame 생성
        asos_df = asos_df.iloc[:, keep_mask]
        
    # 2. 최종 시간 컬럼 이름 확정
    asos_time_col = None
    time_candidates = ['Date&Time', 'date', 'datetime', 'tm', 'time', 'timestamp']
    for col in time_candidates:
        if col in asos_df.columns:
            asos_time_col = col
            break
            
    if asos_time_col is None:
        raise ValueError("ASOS 데이터에서 시간 컬럼을 찾을 수 없습니다.")

    print(f"  ✓ 시간 컬럼 발견: '{asos_time_col}'")
    
    # 3. 시간 컬럼을 'Date&Time'으로 표준화 및 변환
    if asos_time_col != 'Date&Time':
        asos_df = asos_df.rename(columns={asos_time_col: 'Date&Time'})
        
    asos_df['Date&Time'] = pd.to_datetime(asos_df['Date&Time']) 
    print("  ✓ 'Date&Time' 컬럼을 datetime으로 변환 완료.")

    # 4. ASOS 데이터 컬럼 표준화 (기존 코드 유지)
    asos_column_mapping = {
        'ta': 'outer_temp', 'temp': 'outer_temp', 'temperature': 'outer_temp',
        'hm': 'outer_hum', 'rh': 'outer_hum', 'humidity': 'outer_hum',
        'ws': 'wind_speed', 'wind': 'wind_speed',
        'wd': 'wind_dir', 'wind_direction': 'wind_dir',
        'rn': 'rainfall', 'precipitation': 'rainfall', 'rain': 'rainfall',
        'si': 'solar_rad', 'solar': 'solar_rad', 'radiation': 'solar_rad',
        'icsr': 'solar_rad', 'ps': 'pressure',
    }
    
    for old_name, new_name in asos_column_mapping.items():
        if old_name in asos_df.columns:
            asos_df = asos_df.rename(columns={old_name: new_name})
            
    # 온실 데이터의 outer_temp와 radiation을 ASOS 데이터로 대체하지 않기 위해 병합 컬럼 조정
    asos_merge_cols = [col for col in ['Date&Time', 'outer_hum', 'wind_speed', 'wind_dir', 
                                       'rainfall', 'solar_rad', 'pressure'] if col in asos_df.columns]
    
    asos_df = asos_df[asos_merge_cols].drop_duplicates(subset=['Date&Time'], keep='first')
    
    print(f"  ✓ ASOS 데이터: {asos_df.shape}")
    print(f"  ✓ ASOS 컬럼 표준화 완료")

    # 3. 데이터 병합
    print("\n[3] 온실 데이터 + ASOS 기상 데이터 병합")
    
    greenhouse_df = greenhouse_df.sort_values('Date&Time')
    asos_df = asos_df.sort_values('Date&Time')
    
    merged_df = pd.merge_asof(
        greenhouse_df,
        asos_df,
        on='Date&Time',
        direction='nearest',
        tolerance=pd.Timedelta('1H')
    )
    
    # 기상 데이터 결측치 처리 (ASOS에서 가져온 컬럼만 처리)
    weather_cols_from_asos = [col for col in asos_merge_cols if col != 'Date&Time']
    
    for col in weather_cols_from_asos:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
            if merged_df[col].isnull().sum() > 0:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    print(f"  ✓ 병합 완료: {merged_df.shape}")
    
    final_cols_order = ['Date&Time', 'inner_temp', 'inner_hum', 'outer_temp', 
                        'radiation', 'radiation_sum', 'release_cooling', 
                        'meas_lee_vent_contr', 'status_swi_cool', 'vent_temp', 
                        'meas_curtain_2', 'set_heat_temp', 'set_vent_temp', 'heat_temp'] + weather_cols_from_asos
    
    final_cols_order = list(dict.fromkeys(final_cols_order))
    final_cols_order = [col for col in final_cols_order if col in merged_df.columns]
    
    return merged_df[final_cols_order]

# ==============================================================================
# 나머지 함수들 (변경 없음)
# ==============================================================================

def add_time_features(df):
    """시간 관련 특성 추가"""
    print("\n[4] 시간 특성 추가")
    
    df = df.copy()
    
    df['hour'] = df['Date&Time'].dt.hour
    df['day'] = df['Date&Time'].dt.day
    df['month'] = df['Date&Time'].dt.month
    df['dayofweek'] = df['Date&Time'].dt.dayofweek
    
    # 주기적 시간 특성 (sin/cos 인코딩)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # 계절 (0: 겨울, 1: 봄, 2: 여름, 3: 가을)
    df['season'] = df['month'].apply(lambda x: 
        0 if x in [12, 1, 2] else
        1 if x in [3, 4, 5] else
        2 if x in [6, 7, 8] else 3
    )
    
    # 주말 여부
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    print(f"  ✓ 시간 특성 추가 완료: 12개 특성")
    
    return df


def handle_missing_values(df):
    """결측치 처리"""
    print("\n[5] 결측치 처리")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  ⚠️  결측치 발견:")
        for col, count in missing[missing > 0].items():
            print(f"    - {col}: {count}개 ({count/len(df)*100:.2f}%)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            if df[col].isnull().sum() > 0:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(mean_val)
        
        print(f"  ✓ 결측치 처리 완료")
    else:
        print(f"  ✓ 결측치 없음")
    
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
    
    print(f"  ✓ 이상치 처리 완료: {outlier_count}개 값 수정")
    
    return df


def create_lag_features(df, target_cols=['inner_temp', 'inner_hum'], lags=[1, 3, 6]):
    """지연(lag) 특성 생성"""
    print("\n[7] 지연(Lag) 특성 생성")
    print(f"  대상 컬럼: {target_cols}")
    print(f"  Lag 시간: {lags}시간")
    
    df = df.copy()
    lag_cols_created = []
    
    for col in target_cols:
        if col not in df.columns:
            print(f"  ⚠️  컬럼을 찾을 수 없음: {col}")
            continue
        
        for lag in lags:
            lag_col_name = f'{col}_lag_{lag}'
            df[lag_col_name] = df[col].shift(lag)
            lag_cols_created.append(lag_col_name)
            print(f"    ✓ {lag_col_name} 생성")
    
    print(f"  ✓ 생성된 lag 특성: {len(lag_cols_created)}개")
    
    # Lag로 인한 결측치 제거
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"  ✓ Lag 결측치 제거: {removed}개 행")
    
    return df


def preprocess_data(greenhouse_path, asos_weather_path, output_path, 
                    add_lags=True, lag_hours=[1, 3, 6],
                    handle_outlier=True, outlier_method='iqr'):
    """전체 전처리 파이프라인"""
    print("\n" + "="*60)
    print("전처리 파이프라인 시작")
    print("="*60)
    
    # 1. 데이터 로드 및 병합
    df = load_and_merge_data(greenhouse_path, asos_weather_path)
    
    # 2. 시간 특성 추가
    df = add_time_features(df)
    
    # 3. 결측치 처리
    df = handle_missing_values(df)
    
    # 4. 이상치 처리
    if handle_outlier:
        df = handle_outliers(df, method=outlier_method, threshold=3)
    
    # 5. Lag 특성 생성
    if add_lags:
        df = create_lag_features(df, target_cols=['inner_temp', 'inner_hum'], lags=lag_hours)
    
    # 6. 정렬 및 인덱스 재설정
    df = df.sort_values('Date&Time').reset_index(drop=True)
    
    # 7. 최종 결과
    print("\n" + "="*60)
    print("[8] 전처리 완료")
    print("="*60)
    print(f"  ✓ 최종 데이터: {df.shape}")
    print(f"  ✓ 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
    print(f"  ✓ 총 시간: {len(df)}시간")
    print(f"\n  ✓ 컬럼 목록 ({len(df.columns)}개):")
    
    # 컬럼을 카테고리별로 분류
    time_cols = [col for col in df.columns if 'Date&Time' in col]
    inner_cols = [col for col in df.columns if 'inner' in col]
    outer_cols = [col for col in df.columns if 'outer' in col or col in ['wind_speed', 'wind_dir', 'rainfall', 'solar_rad', 'pressure']]
    time_feature_cols = [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'season', 'weekend', 'sin', 'cos'])]
    
    print(f"    - 시간: {time_cols}")
    print(f"    - 온실: {inner_cols}")
    print(f"    - 기상: {outer_cols}")
    print(f"    - 시간특성: {time_feature_cols}")
    
    # 8. 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    # 9. 통계
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
    
    GREENHOUSE_PATH = 'input/greenhouse_inner_8_1year.csv'
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
            lag_hours=[1, 3, 6],  # 1시간, 3시간, 6시간 전 데이터
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