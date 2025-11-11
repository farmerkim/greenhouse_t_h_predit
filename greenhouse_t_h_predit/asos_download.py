# asos_download.py - SSL 오류 수정 버전
import urllib.request
import urllib.parse
import pandas as pd
import os
import ssl
from datetime import datetime, timedelta
import json
import time
import requests  # requests 라이브러리 사용 (더 안정적)

def fetch_weather_data_requests(start_date, end_date, stn_ids):
    """requests 라이브러리를 사용한 데이터 다운로드 (권장)"""
    startDt = start_date.replace('-', '')
    endDt = end_date.replace('-', '')

    url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    all_data = []
    page_no = 1
    total_pages = 1
    max_retries = 3

    while page_no <= total_pages:
        params = {
            'serviceKey': 'c41d42c7c683c85b3e54a9bc00ec9d9e71f052d73a0722a759c14d40836f05cb',
            'numOfRows': '100',
            'pageNo': str(page_no),
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'HR',
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
                print(f"  페이지 {page_no}/{total_pages} 다운로드 중... ({start_date} ~ {end_date})", end='')
                if retry_count > 0:
                    print(f" [재시도 {retry_count}/{max_retries}]", end='')
                print()
                
                # requests 사용 (타임아웃 증가, 연결 재사용)
                response = requests.get(
                    url, 
                    params=params, 
                    verify=False, 
                    timeout=(10, 60),  # (연결 타임아웃, 읽기 타임아웃)
                    headers={'Connection': 'close'}  # 연결 재사용 방지
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'response' not in data:
                        print(f"  ⚠️ 응답 형식 오류")
                        return None
                    
                    # 에러 코드 확인
                    if 'header' in data['response']:
                        result_code = data['response']['header'].get('resultCode', '00')
                        result_msg = data['response']['header'].get('resultMsg', 'SUCCESS')
                        
                        if result_code != '00':
                            print(f"  ❌ API 오류: [{result_code}] {result_msg}")
                            return None
                    
                    if 'body' not in data['response'] or 'items' not in data['response']['body']:
                        print(f"  ⚠️ 데이터 없음: {start_date} ~ {end_date}")
                        return None

                    items = data['response']['body']['items']['item']
                    
                    # 단일 항목인 경우 리스트로 변환
                    if isinstance(items, dict):
                        items = [items]
                    
                    df = pd.json_normalize(items)
                    all_data.append(df)
                    
                    print(f"  ✓ {len(df)} 레코드 다운로드 완료")

                    # 첫 페이지에서 전체 페이지 수 계산
                    if page_no == 1:
                        total_count = data['response']['body']['totalCount']
                        total_pages = (total_count // 100) + (1 if total_count % 100 > 0 else 0)
                        print(f"  총 {total_count} 레코드, {total_pages} 페이지")

                    page_no += 1
                    success = True
                    time.sleep(1)  # API 호출 간격 증가
                    
                elif response.status_code == 500:
                    print(f"  ⚠️ 서버 오류 (500), 재시도...")
                    retry_count += 1
                    time.sleep(3)
                else:
                    print(f"  ❌ HTTP 오류: {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"  ⏱️ 타임아웃 발생, {3 * retry_count}초 후 재시도...")
                time.sleep(3 * retry_count)
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 요청 오류: {e}")
                retry_count += 1
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ 예상치 못한 오류: {e}")
                return None
        
        if not success:
            print(f"  ❌ {max_retries}번 재시도 실패")
            return None

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"  ✅ 총 {len(final_df)} 레코드 수집 완료")
        return final_df
    else:
        return None


def fetch_weather_data_urllib(start_date, end_date, stn_ids):
    """urllib를 사용한 데이터 다운로드 (대체 방법)"""
    startDt = start_date.replace('-', '')
    endDt = end_date.replace('-', '')

    url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    # SSL 컨텍스트 생성 (검증 비활성화)
    ssl_context = ssl._create_unverified_context()

    all_data = []
    page_no = 1
    total_pages = 1

    while page_no <= total_pages:
        params = {
            'serviceKey': 'c41d42c7c683c85b3e54a9bc00ec9d9e71f052d73a0722a759c14d40836f05cb',
            'numOfRows': '720',
            'pageNo': str(page_no),
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'HR',
            'startDt': startDt,
            'startHh': '00',
            'endDt': endDt,
            'endHh': '23',
            'stnIds': stn_ids
        }

        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        try:
            print(f"  페이지 {page_no}/{total_pages} 다운로드 중...")
            
            with urllib.request.urlopen(full_url, context=ssl_context, timeout=30) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    
                    if 'body' not in data['response'] or 'items' not in data['response']['body']:
                        print(f"  ⚠️ 데이터 없음")
                        return None

                    items = data['response']['body']['items']['item']
                    
                    if isinstance(items, dict):
                        items = [items]
                    
                    df = pd.json_normalize(items)
                    all_data.append(df)
                    
                    print(f"  ✓ {len(df)} 레코드 다운로드 완료")

                    if page_no == 1:
                        total_count = data['response']['body']['totalCount']
                        total_pages = (total_count // 720) + (1 if total_count % 720 > 0 else 0)

                    page_no += 1
                    time.sleep(0.5)
                else:
                    print(f"  ❌ HTTP 오류: {response.status}")
                    break
                    
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            break

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        return None


def save_data(df, region_code, cache_dir):
    """데이터 저장"""
    df['year'] = pd.to_datetime(df['tm']).dt.year.astype(str)
    df['month'] = pd.to_datetime(df['tm']).dt.month.astype(str).str.zfill(2)

    for (year, month), group in df.groupby(['year', 'month']):
        year_month_dir = os.path.join(cache_dir, str(region_code), year)
        os.makedirs(year_month_dir, exist_ok=True)
        filename = os.path.join(year_month_dir, f"{month}.csv")
        group.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  저장: {filename} ({len(group)} 레코드)")


def process_asos_data(asos):
    """ASOS 데이터 전처리"""
    asos = asos.copy()
    asos['일시'] = pd.to_datetime(asos['tm'], format='%Y-%m-%d %H:%M')
    asos['날짜'] = asos['일시'].dt.date
    asos['시간'] = asos['일시'].dt.strftime('%H:%M')
    asos['icsr'] = asos['icsr'].replace('', 0)
    
    # 필요한 컬럼만 선택
    asos = asos[['stnNm', '날짜', '시간', 'icsr', 'ta', 'ws', 'hm', 'tm']]
    asos.rename(columns={
        'icsr': '일사(MJ/m2)', 
        'stnNm': '지점', 
        'ta': '온도', 
        'ws': '풍속', 
        'hm': '습도'
    }, inplace=True)
    
    return asos


def main():
    print("="*60)
    print("ASOS 기상 데이터 다운로드")
    print("="*60)
    
    # 설정
    start_date = '2025-04-16'
    end_date = '2025-10-27'
    stn_ids = '146'  # 전주 기상대
    
    output_dir = 'output'
    cache_dir = os.path.join(output_dir, 'cache', 'ASOS')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n시작일: {start_date}")
    print(f"종료일: {end_date}")
    print(f"지점: {stn_ids} (전주)")
    print(f"저장 경로: {cache_dir}")
    
    # API 키 테스트
    print("\n🔍 API 연결 테스트 중...")
    test_url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    test_params = {
        'serviceKey': 'c41d42c7c683c85b3e54a9bc00ec9d9e71f052d73a0722a759c14d40836f05cb',
        'numOfRows': '1',
        'pageNo': '1',
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'HR',
        'startDt': '20250416',
        'startHh': '00',
        'endDt': '20251026',
        'endHh': '23',
        'stnIds': stn_ids
    }
    
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        test_response = requests.get(test_url, params=test_params, verify=False, timeout=10)
        if test_response.status_code == 200:
            test_data = test_response.json()
            if 'response' in test_data and 'header' in test_data['response']:
                result_code = test_data['response']['header'].get('resultCode', '00')
                result_msg = test_data['response']['header'].get('resultMsg', 'SUCCESS')
                
                if result_code == '00':
                    print("✅ API 연결 성공!")
                else:
                    print(f"⚠️ API 오류: [{result_code}] {result_msg}")
                    print("\n가능한 원인:")
                    print("1. API 키가 만료되었거나 잘못되었습니다")
                    print("2. 일일 트래픽 제한을 초과했습니다")
                    print("3. 해당 날짜에 데이터가 없습니다")
                    return
        else:
            print(f"⚠️ HTTP 오류: {test_response.status_code}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("\n대체 방법:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. VPN을 사용 중이라면 비활성화하세요")
        print("3. 방화벽 설정을 확인하세요")
        return
    
    print()
    
    # requests 사용
    use_requests = True
    print("✓ requests 라이브러리 사용 (타임아웃: 60초, 재시도: 3회)\n")
    
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    current_date = start_date_obj
    success_count = 0
    fail_count = 0
    
    # 기간을 더 작게 분할 (주 단위)
    while current_date <= end_date_obj:
        # 1주일 단위로 분할 (타임아웃 방지)
        fetch_end_date = min(current_date + timedelta(days=6), end_date_obj)
        
        print(f"\n[{current_date.strftime('%Y-%m-%d')} ~ {fetch_end_date.strftime('%Y-%m-%d')}]")
        
        try:
            # requests 사용 (우선)
            if use_requests:
                asos_df = fetch_weather_data_requests(
                    current_date.strftime('%Y-%m-%d'), 
                    fetch_end_date.strftime('%Y-%m-%d'), 
                    stn_ids
                )
            else:
                asos_df = fetch_weather_data_urllib(
                    current_date.strftime('%Y-%m-%d'), 
                    fetch_end_date.strftime('%Y-%m-%d'), 
                    stn_ids
                )
            
            if asos_df is not None:
                asos_df = process_asos_data(asos_df)
                save_data(asos_df, stn_ids, cache_dir)
                success_count += 1
            else:
                print(f"  ⚠️ 데이터 없음")
                fail_count += 1
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            fail_count += 1
        
        current_date += timedelta(days=7)
        time.sleep(1)  # API 호출 간격
    
    print("\n" + "="*60)
    print("다운로드 완료")
    print("="*60)
    print(f"성공: {success_count}개월")
    print(f"실패: {fail_count}개월")
    print(f"저장 경로: {cache_dir}")
    
    # 다운로드된 파일 확인
    csv_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"다운로드된 파일: {len(csv_files)}개")


if __name__ == "__main__":
    main()