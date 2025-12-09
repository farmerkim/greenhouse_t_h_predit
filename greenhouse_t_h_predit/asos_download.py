import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_weather_data(start_date, end_date, stn_ids, service_key):
    """
    ASOS 기상 데이터 다운로드
    
    Args:
        start_date (str): 시작일 (YYYY-MM-DD)
        end_date (str): 종료일 (YYYY-MM-DD)
        stn_ids (str): 기상대 코드
        service_key (str): API 인증키
    Returns:
        pd.DataFrame: 기상 데이터
    """
    startDt = start_date.replace('-', '')
    endDt = end_date.replace('-', '')

    url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    all_data = []
    page_no = 1
    total_pages = 1
    max_retries = 3

    print(f"\n📥 데이터 다운로드 시작: {start_date} ~ {end_date}")
    
    while page_no <= total_pages:
        params = {
            'serviceKey': service_key,
            'numOfRows': '999',  # 한 페이지당 최대 레코드 수
            'pageNo': str(page_no),
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'HR',  # 시간별 데이터
            'startDt': startDt,
            'startHh': '00',
            'endDt': endDt,
            'endHh': '23',
            'stnIds': stn_ids
        }

        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"  📄 페이지 {page_no}/{total_pages} 다운로드 중...", end='')
                if retry_count > 0:
                    print(f" [재시도 {retry_count}/{max_retries}]", end='')
                print()
                
                response = requests.get(
                    url, 
                    params=params, 
                    verify=False, 
                    timeout=(10, 60),
                    headers={'Connection': 'close'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 응답 확인
                    if 'response' not in data:
                        print(f"    ⚠️ 응답 형식 오류")
                        return None
                    
                    # 에러 코드 확인
                    if 'header' in data['response']:
                        result_code = data['response']['header'].get('resultCode', '00')
                        result_msg = data['response']['header'].get('resultMsg', 'SUCCESS')
                        
                        if result_code != '00':
                            print(f"    ❌ API 오류: [{result_code}] {result_msg}")
                            if result_code == '03':
                                print(f"    → 해당 기간에 데이터가 없습니다.")
                            return None
                    
                    # 데이터 확인
                    if 'body' not in data['response'] or 'items' not in data['response']['body']:
                        print(f"    ⚠️ 데이터 없음")
                        return None

                    items = data['response']['body']['items']['item']
                    
                    # 단일 항목인 경우 리스트로 변환
                    if isinstance(items, dict):
                        items = [items]
                    
                    df = pd.json_normalize(items)
                    all_data.append(df)
                    
                    print(f"    ✓ {len(df)} 레코드 다운로드 완료")

                    # 첫 페이지에서 전체 페이지 수 계산
                    if page_no == 1:
                        total_count = data['response']['body']['totalCount']
                        num_of_rows = int(params['numOfRows'])
                        total_pages = (total_count // num_of_rows) + (1 if total_count % num_of_rows > 0 else 0)
                        print(f"    📊 총 {total_count} 레코드, {total_pages} 페이지")

                    page_no += 1
                    success = True
                    time.sleep(0.5)  # API 호출 간격
                    
                elif response.status_code == 500:
                    print(f"    ⚠️ 서버 오류 (500), 재시도...")
                    retry_count += 1
                    time.sleep(3)
                else:
                    print(f"    ❌ HTTP 오류: {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"    ⏱️ 타임아웃 발생, {3 * retry_count}초 후 재시도...")
                time.sleep(3 * retry_count)
                
            except requests.exceptions.RequestException as e:
                print(f"    ❌ 요청 오류: {e}")
                retry_count += 1
                time.sleep(2)
                
            except Exception as e:
                print(f"    ❌ 예상치 못한 오류: {e}")
                return None
        
        if not success:
            print(f"    ❌ {max_retries}번 재시도 실패")
            return None

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ 총 {len(final_df)} 레코드 수집 완료")
        return final_df
    else:
        return None


def process_asos_data(df):
    """
    ASOS 데이터 전처리
    
    Args:
        df (pd.DataFrame): 원본 ASOS 데이터
    Returns:
        pd.DataFrame: 전처리된 데이터
    """
    print("\n🔧 데이터 전처리 중...")
    
    df = df.copy()
    
    # 기존 컬럼 확인
    print(f"  원본 컬럼: {list(df.columns)}")
    
    # 컬럼 매핑 (ASOS 원본 → 표준명)
    column_mapping = {
        'tm': 'datetime_temp',   # 임시로 다른 이름 사용
        'ta': 'outer_temp',      # 기온(℃)
        'hm': 'outer_hum',       # 습도(%)
        'ws': 'wind_speed',      # 풍속(m/s)
        'wd': 'wind_dir',        # 풍향(deg)
        'rn': 'rainfall',        # 강수량(mm)
        'icsr': 'solar_rad',     # 일사(MJ/m2)
        'ps': 'pressure'         # 현지기압(hPa)
    }
    
    # 존재하는 컬럼만 이름 변경
    cols_to_rename = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            cols_to_rename[old_col] = new_col
    
    df = df.rename(columns=cols_to_rename)
    
    # datetime_temp를 Date&Time으로 변환
    if 'datetime_temp' in df.columns:
        df['Date&Time'] = pd.to_datetime(df['datetime_temp'], format='%Y-%m-%d %H:%M')
        df = df.drop('datetime_temp', axis=1)
    else:
        print(f"  ⚠️ 시간 컬럼(tm)을 찾을 수 없습니다.")
        return None
    
    # 필요한 컬럼만 선택
    cols_to_keep = ['Date&Time']
    for col in ['outer_temp', 'outer_hum', 'wind_speed', 'wind_dir', 
                'rainfall', 'solar_rad', 'pressure']:
        if col in df.columns:
            cols_to_keep.append(col)
    
    df = df[cols_to_keep]
    
    # 데이터 타입 변환 및 결측치 처리
    numeric_cols = [col for col in df.columns if col != 'Date&Time']
    for col in numeric_cols:
        # 숫자로 변환
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # -99 같은 결측치 표시를 NaN으로
        df.loc[df[col] < -90, col] = None
    
    # 정렬
    df = df.sort_values('Date&Time').reset_index(drop=True)
    
    # 중복 제거
    df = df.drop_duplicates(subset=['Date&Time'], keep='first')
    
    print(f"  ✓ 전처리 완료")
    print(f"  ✓ 컬럼: {list(df.columns)}")
    print(f"  ✓ 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
    print(f"  ✓ 레코드 수: {len(df)}")
    
    return df


def test_api_connection(service_key, stn_ids):
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")
    
    test_url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    test_params = {
        'serviceKey': service_key,
        'numOfRows': '1',
        'pageNo': '1',
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'HR',
        'startDt': '20250416',
        'startHh': '00',
        'endDt': '20250416',
        'endHh': '01',
        'stnIds': stn_ids
    }
    
    try:
        response = requests.get(test_url, params=test_params, verify=False, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and 'header' in data['response']:
                result_code = data['response']['header'].get('resultCode', '00')
                result_msg = data['response']['header'].get('resultMsg', 'SUCCESS')
                
                if result_code == '00':
                    print("  ✅ API 연결 성공!")
                    return True
                else:
                    print(f"  ⚠️ API 오류: [{result_code}] {result_msg}")
                    print("\n  가능한 원인:")
                    print("    1. API 키가 만료되었거나 잘못되었습니다")
                    print("    2. 일일 트래픽 제한을 초과했습니다")
                    print("    3. 해당 날짜에 데이터가 없습니다")
                    return False
        else:
            print(f"  ⚠️ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ 테스트 실패: {e}")
        print("\n  대체 방법:")
        print("    1. 인터넷 연결을 확인하세요")
        print("    2. VPN을 사용 중이라면 비활성화하세요")
        print("    3. 방화벽 설정을 확인하세요")
        return False


def download_asos_data_by_period(start_date, end_date, stn_ids, service_key, days_per_fetch=30):
    """
    기간을 나누어 ASOS 데이터 다운로드
    
    Args:
        start_date (str): 시작일
        end_date (str): 종료일
        stn_ids (str): 기상대 코드
        service_key (str): API 인증키
        days_per_fetch (int): 한 번에 가져올 일수
    Returns:
        pd.DataFrame: 통합 데이터
    """
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_date = start_date_obj
    
    while current_date <= end_date_obj:
        fetch_end_date = min(current_date + timedelta(days=days_per_fetch-1), end_date_obj)
        
        print(f"\n{'='*60}")
        print(f"📅 기간: {current_date.strftime('%Y-%m-%d')} ~ {fetch_end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        try:
            df = fetch_weather_data(
                current_date.strftime('%Y-%m-%d'),
                fetch_end_date.strftime('%Y-%m-%d'),
                stn_ids,
                service_key
            )
            
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"  ✅ {len(df)} 레코드 수집")
            else:
                print(f"  ⚠️ 데이터 없음")
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
        
        current_date = fetch_end_date + timedelta(days=1)
        time.sleep(1)  # API 호출 간격
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # tm 컬럼 기준으로 중복 제거 (Date&Time으로 변환 전)
        if 'tm' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['tm'], keep='first')
        return combined_df
    else:
        return None


def main():
    print("="*60)
    print("🌤️  ASOS 기상 데이터 다운로드")
    print("="*60)
    
    # ========== 설정 ==========
    START_DATE = '2024-12-01'  # 시작일
    END_DATE = '2025-12-01'    # 종료일
    STN_IDS = '146'            # 전주 기상대 (146)
    SERVICE_KEY = ''
    
    OUTPUT_DIR = 'input'       # 저장 폴더
    OUTPUT_FILE = 'asos_weather.csv'  # 저장 파일명
    # ==========================
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"\n📋 다운로드 설정:")
    print(f"  • 시작일: {START_DATE}")
    print(f"  • 종료일: {END_DATE}")
    print(f"  • 기상대: {STN_IDS} (전주)")
    print(f"  • 저장 경로: {output_path}")
    
    # 기간 계산
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days + 1
    total_hours = total_days * 24
    
    print(f"  • 기간: {total_days}일 ({total_hours}시간)")
    
    # API 연결 테스트
    if not test_api_connection(SERVICE_KEY, STN_IDS):
        print("\n❌ API 연결 실패. 프로그램을 종료합니다.")
        return
    
    # 데이터 다운로드
    print("\n" + "="*60)
    print("📥 데이터 다운로드 시작")
    print("="*60)
    
    # 30일 단위로 나누어 다운로드 (API 타임아웃 방지)
    asos_df = download_asos_data_by_period(
        START_DATE, 
        END_DATE, 
        STN_IDS, 
        SERVICE_KEY,
        days_per_fetch=30
    )
    
    if asos_df is None or len(asos_df) == 0:
        print("\n❌ 데이터를 다운로드할 수 없습니다.")
        return
    
    # 데이터 전처리
    asos_df = process_asos_data(asos_df)
    
    if asos_df is None:
        print("\n❌ 데이터 전처리에 실패했습니다.")
        return
    
    # 데이터 저장
    print(f"\n💾 데이터 저장 중...")
    asos_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 저장 완료: {output_path}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("✅ 다운로드 완료!")
    print("="*60)
    print(f"📊 데이터 요약:")
    print(f"  • 총 레코드: {len(asos_df)}개")
    print(f"  • 기간: {asos_df['Date&Time'].min()} ~ {asos_df['Date&Time'].max()}")
    print(f"  • 컬럼: {list(asos_df.columns)}")
    print(f"  • 저장 경로: {output_path}")
    
    # 통계 정보
    print(f"\n📈 기상 데이터 통계:")
    stats_cols = []
    for col in ['outer_temp', 'outer_hum', 'wind_speed', 'rainfall']:
        if col in asos_df.columns:
            stats_cols.append(col)
    
    if stats_cols:
        print(asos_df[stats_cols].describe())
    
    # 결측치 확인
    missing = asos_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  결측치:")
        for col, count in missing[missing > 0].items():
            print(f"  • {col}: {count}개 ({count/len(asos_df)*100:.1f}%)")
    else:
        print(f"\n✅ 결측치 없음")
    
    print("\n" + "="*60)
    print("="*60)


if __name__ == "__main__":

    main()
