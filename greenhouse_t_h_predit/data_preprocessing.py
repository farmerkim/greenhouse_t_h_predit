# data_preprocessing.py - 인코딩 오류 수정 버전
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def preprocess_env(env_dir):
    """외부 환경 데이터 전처리"""
    csv_files = []

    for root, dirs, files in os.walk(env_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"ASOS 데이터를 찾을 수 없습니다: {env_dir}")

    print(f"발견된 ASOS 파일: {len(csv_files)}개")

    df_list = []
    for file in csv_files:
        try:
            # 여러 인코딩 시도
            try:
                df = pd.read_csv(file, parse_dates=['tm'], encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(file, parse_dates=['tm'], encoding='cp949')
            except:
                df = pd.read_csv(file, parse_dates=['tm'], encoding='euc-kr')
            
            df_list.append(df)
            print(f"  ✓ {os.path.basename(file)}: {len(df)} 레코드")
        except Exception as e:
            print(f"  ✗ {os.path.basename(file)}: {e}")

    if not df_list:
        raise ValueError("읽을 수 있는 ASOS 파일이 없습니다.")

    final_df = pd.concat(df_list, ignore_index=True)
    
    # 중복 제거 및 정렬
    final_df = final_df.drop_duplicates(subset=['tm']).sort_values('tm').reset_index(drop=True)

    final_df['year'] = final_df['tm'].dt.year
    final_df['month'] = final_df['tm'].dt.month
    final_df['day'] = final_df['tm'].dt.day
    final_df['hour'] = final_df['tm'].dt.hour

    print(f"전체 ASOS 데이터: {len(final_df)} 레코드")
    print(f"기간: {final_df['tm'].min()} ~ {final_df['tm'].max()}")

    return final_df


def preprocess_inner(inner_path):
    """온실 내부 환경 데이터 전처리"""
    print(f"\n온실 데이터 로드 중: {inner_path}")
    
    # 여러 인코딩 시도
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']
    inner = None
    
    for encoding in encodings:
        try:
            # 빈 컬럼 무시하고 읽기
            inner = pd.read_csv(inner_path, encoding=encoding, skipinitialspace=True)
            
            # 이름이 없거나 빈 컬럼 제거
            inner = inner.loc[:, ~inner.columns.str.contains('^Unnamed')]
            inner = inner.dropna(axis=1, how='all')  # 모든 값이 NaN인 컬럼 제거
            
            print(f"  ✓ 인코딩: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  ✗ {encoding}: {e}")
    
    if inner is None:
        raise ValueError(f"파일을 읽을 수 없습니다: {inner_path}")
    
    print(f"  컬럼: {list(inner.columns)}")
    print(f"  레코드 수: {len(inner)}")
    
    # Date&Time 컬럼 찾기
    date_col = None
    for col in inner.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        # 첫 번째 컬럼을 날짜로 가정
        date_col = inner.columns[0]
        print(f"  ⚠️ 날짜 컬럼 자동 선택: {date_col}")
    else:
        print(f"  날짜/시간 컬럼: {date_col}")
    
    # 날짜 파싱 (MM/DD/YYYY HH:MM 형식)
    inner[date_col] = pd.to_datetime(inner[date_col], format='%m/%d/%Y %H:%M', errors='coerce')
    
    # 파싱 실패한 행 제거
    invalid_dates = inner[date_col].isna().sum()
    if invalid_dates > 0:
        print(f"  ⚠️ 날짜 파싱 실패: {invalid_dates}개 행 제거")
        inner = inner.dropna(subset=[date_col])
    
    # Date&Time이 아닌 경우 컬럼명 변경
    if date_col != 'Date&Time':
        inner = inner.rename(columns={date_col: 'Date&Time'})
    
    # 온도와 습도 컬럼 확인 및 정리
    temp_col = None
    hum_col = None
    
    for col in inner.columns:
        col_lower = col.lower()
        if 'temp' in col_lower and temp_col is None:
            temp_col = col
        elif ('rh' in col_lower or 'hum' in col_lower) and hum_col is None:
            hum_col = col
    
    if temp_col is None or hum_col is None:
        print(f"  ⚠️ 온도/습도 컬럼을 찾을 수 없습니다.")
        print(f"     현재 컬럼: {list(inner.columns)}")
        # 두 번째, 세 번째 컬럼을 온도, 습도로 가정
        if len(inner.columns) >= 3:
            temp_col = inner.columns[1]
            hum_col = inner.columns[2]
            print(f"  자동 선택 - 온도: {temp_col}, 습도: {hum_col}")
    else:
        print(f"  온도 컬럼: {temp_col}")
        print(f"  습도 컬럼: {hum_col}")
    
    # 필요한 컬럼만 선택
    inner = inner[['Date&Time', temp_col, hum_col]].copy()
    
    # 컬럼명 통일
    inner = inner.rename(columns={
        temp_col: 'Air_Temp_Avg',
        hum_col: 'RH_Avg'
    })
    
    # 숫자형으로 변환 (잘못된 값 제거)
    inner['Air_Temp_Avg'] = pd.to_numeric(inner['Air_Temp_Avg'], errors='coerce')
    inner['RH_Avg'] = pd.to_numeric(inner['RH_Avg'], errors='coerce')
    
    # 🚨 임계값 처리 (이상치 제거)
    # 온실 내부 온도의 현실적 범위 (예: 50°C 초과는 이상치로 간주)
    inner.loc[inner['Air_Temp_Avg'] > 40, 'Air_Temp_Avg'] = np.nan 
    # 습도의 현실적 범위 (100% 초과는 이상치로 간주, 1000%는 확실한 오류)
    inner.loc[inner['RH_Avg'] > 95, 'RH_Avg'] = np.nan

    # NaN 제거
    before_len = len(inner)
    inner = inner.dropna()
    after_len = len(inner)
    
    if before_len != after_len:
        print(f"  정리: {before_len - after_len}개 행 제거 (결측치/이상값)")
    
    inner['year'] = inner['Date&Time'].dt.year
    inner['month'] = inner['Date&Time'].dt.month
    inner['day'] = inner['Date&Time'].dt.day
    inner['hour'] = inner['Date&Time'].dt.hour
    
    print(f"  최종 레코드: {len(inner)}개")
    print(f"  기간: {inner['Date&Time'].min()} ~ {inner['Date&Time'].max()}")
    print(f"  온도 범위: {inner['Air_Temp_Avg'].min():.1f}°C ~ {inner['Air_Temp_Avg'].max():.1f}°C")
    print(f"  습도 범위: {inner['RH_Avg'].min():.1f}% ~ {inner['RH_Avg'].max():.1f}%")

    return inner


def data_preprocessing(env_dir, inner_path):
    """데이터 전처리 메인 함수"""
    print("="*60)
    print("데이터 전처리 시작")
    print("="*60)
    
    # 외부 환경 데이터
    env = preprocess_env(env_dir)
    
    # 온실 내부 데이터
    inner = preprocess_inner(inner_path)
    
    print("\n데이터 병합 중...")
    
    # 병합
    data = pd.merge(inner, env, on=['year', 'month', 'day', 'hour'], how='left')
    
    print(f"병합 후: {len(data)} 레코드")
    
    # 불필요한 컬럼 제거
    drop_cols = ['year', 'month', 'day', 'hour', 'tm']
    
    # 존재하는 컬럼만 제거
    existing_drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=existing_drop_cols)
    
    # 추가 제거할 컬럼 (있는 경우만)
    optional_drop_cols = ['STEMP', 'SWAT', 'SEC', '지점', '날짜', '시간']
    for col in optional_drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    print(f"\n현재 컬럼: {list(data.columns)}")
    
    # 컬럼명 변경 (존재하는 경우만)
    rename_dict = {}
    
    # 온실 내부 데이터 (이미 Air_Temp_Avg, RH_Avg로 통일됨)
    if 'TEMP' in data.columns:
        rename_dict['TEMP'] = 'inner_temp'
    if 'Air_Temp_Avg' in data.columns:
        rename_dict['Air_Temp_Avg'] = 'inner_temp'
    
    if 'HUMI' in data.columns:
        rename_dict['HUMI'] = 'inner_hum'
    if 'RH_Avg' in data.columns:
        rename_dict['RH_Avg'] = 'inner_hum'
    
    if 'CO2' in data.columns:
        rename_dict['CO2'] = 'inner_CO2'
    if 'PPF' in data.columns:
        rename_dict['PPF'] = 'inner_PPF'
    
    # 외부 환경 데이터
    if '일사(MJ/m2)' in data.columns:
        rename_dict['일사(MJ/m2)'] = 'out_radn_m'
    if '온도' in data.columns:
        rename_dict['온도'] = 'out_temp'
    if '풍속' in data.columns:
        rename_dict['풍속'] = 'out_wind'
    if '습도' in data.columns:
        rename_dict['습도'] = 'out_hum'
    
    if rename_dict:
        data = data.rename(columns=rename_dict)
        print(f"\n컬럼명 변경:")
        for old, new in rename_dict.items():
            print(f"  {old} -> {new}")
    
    # 필수 컬럼 확인
    required_cols = ['Date&Time', 'inner_temp', 'inner_hum']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"\n⚠️ 필수 컬럼 누락: {missing_cols}")
        print(f"현재 컬럼: {list(data.columns)}")
    else:
        print(f"\n✓ 필수 컬럼 확인 완료: inner_temp, inner_hum")
    
    # 일사량 변환 (MJ/m2 -> W/m2)
    if 'out_radn_m' in data.columns:
        data['out_radn_w'] = data['out_radn_m'] * 277.78
        print("일사량 변환: MJ/m2 -> W/m2")
    
    # 결측치 보간
    print(f"\n결측치 처리 전:")
    print(data.isnull().sum()[data.isnull().sum() > 0])
    
    data = data.interpolate(method='linear', limit_direction='both')
    
    # 여전히 남은 NaN은 평균값으로 채우기
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mean(), inplace=True)
    
    print(f"\n결측치 처리 후:")
    remaining_na = data.isnull().sum()[data.isnull().sum() > 0]
    if len(remaining_na) > 0:
        print(remaining_na)
    else:
        print("  결측치 없음")
    
    print(f"\n최종 데이터:")
    print(f"  레코드 수: {len(data)}")
    print(f"  컬럼 수: {len(data.columns)}")
    print(f"  컬럼: {list(data.columns)}")
    
    return data


def plot_data(data):
    """데이터 시각화"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 온도 비교
    if 'out_temp' in data.columns:
        sns.lineplot(data=data, x='Date&Time', y='out_temp', label='외부 온도', ax=ax)
    if 'inner_temp' in data.columns:
        sns.lineplot(data=data, x='Date&Time', y='inner_temp', label='내부 온도', ax=ax)
    
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('날짜/시간')
    ax.set_ylabel('온도 (°C)')
    ax.set_title('온실 내외부 온도 비교')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/temperature_comparison.png', dpi=300, bbox_inches='tight')
    print("\n그래프 저장: output/temperature_comparison.png")
    plt.close()


def main():
    env_dir = 'output/cache/ASOS/146'
    inner_path = 'input/greenhouse_inner.csv'
    
    # 파일 존재 확인
    if not os.path.exists(inner_path):
        print(f"❌ 온실 데이터 파일을 찾을 수 없습니다: {inner_path}")
        print("\n다음을 확인하세요:")
        print("1. input/greenhouse_inner.csv 파일이 존재하는지")
        print("2. 파일 경로가 올바른지")
        return
    
    if not os.path.exists(env_dir):
        print(f"❌ ASOS 데이터 디렉토리를 찾을 수 없습니다: {env_dir}")
        print("\n다음을 확인하세요:")
        print("1. python asos_download.py 를 먼저 실행했는지")
        print("2. output/cache/ASOS/146/ 디렉토리에 CSV 파일이 있는지")
        return

    # 전처리 실행
    data = data_preprocessing(env_dir, inner_path)
    
    # 저장
    output_path = 'output/preprocessed_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 전처리 완료! 저장 위치: {output_path}")
    
    # 데이터 정보 출력
    print("\n" + "="*60)
    print("데이터 정보")
    print("="*60)
    print(data.info())
    
    print("\n" + "="*60)
    print("데이터 미리보기 (첫 5행)")
    print("="*60)
    print(data.head())
    
    # 그래프 생성 (선택)
    try:
        plot_data(data)
    except Exception as e:
        print(f"\n⚠️ 그래프 생성 실패: {e}")


if __name__ == '__main__':
    main()