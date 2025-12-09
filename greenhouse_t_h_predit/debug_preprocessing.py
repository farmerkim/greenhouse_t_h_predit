import pandas as pd

print("="*60)
print("디버깅: 온실 데이터 로드 테스트")
print("="*60)

# 1. 파일 읽기 테스트
print("\n[1] CSV 파일 읽기...")
greenhouse_path = 'input/greenhouse_inner_8_1year.csv'

# 구분자 확인
with open(greenhouse_path, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    delimiter = ';' if ';' in first_line else ','
    print(f"구분자: {delimiter}")

# 3줄 헤더로 읽기
try:
    df = pd.read_csv(greenhouse_path, sep=delimiter, header=[0, 1, 2], encoding='utf-8')
    df.columns = ['_'.join(col).strip().replace(' ', '_').lower() for col in df.columns]
    print(f"✓ 데이터 shape: {df.shape}")
    print(f"✓ 컬럼 수: {len(df.columns)}")
except Exception as e:
    print(f"❌ 읽기 실패: {e}")
    exit()

# 2. 빈 컬럼 제거
print("\n[2] 빈 컬럼 제거 전후...")
print(f"제거 전: {df.shape}")
df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
print(f"제거 후: {df.shape}")
print(f"남은 컬럼: {list(df.columns)}")

# 3. 시간 컬럼 찾기
print("\n[3] 시간 컬럼 처리...")
time_col = df.columns[0]  # 첫 번째 컬럼
print(f"시간 컬럼: '{time_col}'")
print(f"첫 5개 값:\n{df[time_col].head()}")

# 날짜 변환 시도
try:
    df['Date&Time'] = pd.to_datetime(df[time_col], format='%d-%m-%Y %H:%M:%S')
    print(f"✓ 날짜 변환 성공 (DD-MM-YYYY)")
except:
    try:
        df['Date&Time'] = pd.to_datetime(df[time_col], dayfirst=True)
        print(f"✓ 날짜 변환 성공 (자동 감지)")
    except Exception as e:
        print(f"❌ 날짜 변환 실패: {e}")
        exit()

print(f"날짜 범위: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
print(f"데이터 개수: {len(df)}")

# 4. 온도 컬럼 찾기
print("\n[4] 온도 컬럼 찾기...")
temp_cols = [col for col in df.columns if 'temp' in col.lower()]
print(f"온도 관련 컬럼: {temp_cols}")

# grh temp 컬럼 찾기
grh_temp_cols = [col for col in temp_cols if 'grh' in col.lower() and 'meas' in col.lower()]
print(f"온실 온도 후보: {grh_temp_cols}")

if grh_temp_cols:
    temp_col = [col for col in grh_temp_cols if '.1' not in col and '.2' not in col][0] if any('.1' not in col for col in grh_temp_cols) else grh_temp_cols[0]
    print(f"선택된 온도 컬럼: '{temp_col}'")
    
    df['inner_temp'] = df[temp_col]
    print(f"\n온도 데이터 샘플:")
    print(df[['Date&Time', 'inner_temp']].head(10))
    print(f"\n온도 통계:")
    print(df['inner_temp'].describe())
else:
    print("❌ 온실 온도 컬럼을 찾을 수 없습니다!")
    exit()

# 5. 리샘플링 테스트
print("\n[5] 리샘플링 테스트...")
print(f"리샘플링 전: {df.shape}")
print(f"시간 간격 예시: {df['Date&Time'].diff().value_counts().head()}")

df_resampled = df.set_index('Date&Time')[['inner_temp']].resample('1H').mean().reset_index()
print(f"리샘플링 후: {df_resampled.shape}")
print(f"\n리샘플링 후 샘플:")
print(df_resampled.head(10))
print(f"\n결측치: {df_resampled['inner_temp'].isnull().sum()}개")

# 6. ASOS 데이터 확인
print("\n[6] ASOS 데이터 확인...")
asos_path = 'input/asos_weather.csv'
try:
    asos_df = pd.read_csv(asos_path)
    print(f"✓ ASOS 데이터: {asos_df.shape}")
    print(f"컬럼: {list(asos_df.columns)}")
    
    # 시간 컬럼 찾기
    time_cols = [col for col in asos_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        asos_time_col = time_cols[0]
        asos_df['Date&Time'] = pd.to_datetime(asos_df[asos_time_col])
        print(f"ASOS 날짜 범위: {asos_df['Date&Time'].min()} ~ {asos_df['Date&Time'].max()}")
        
        # 날짜 범위 겹치는지 확인
        greenhouse_start = df_resampled['Date&Time'].min()
        greenhouse_end = df_resampled['Date&Time'].max()
        asos_start = asos_df['Date&Time'].min()
        asos_end = asos_df['Date&Time'].max()
        
        print(f"\n날짜 범위 비교:")
        print(f"  온실: {greenhouse_start} ~ {greenhouse_end}")
        print(f"  ASOS: {asos_start} ~ {asos_end}")
        
        if greenhouse_start > asos_end or greenhouse_end < asos_start:
            print(f"  ⚠️  날짜 범위가 겹치지 않습니다!")
        else:
            overlap_start = max(greenhouse_start, asos_start)
            overlap_end = min(greenhouse_end, asos_end)
            print(f"  ✓ 겹치는 구간: {overlap_start} ~ {overlap_end}")
            
except Exception as e:
    print(f"❌ ASOS 데이터 오류: {e}")

print("\n" + "="*60)
print("디버깅 완료")
print("="*60)