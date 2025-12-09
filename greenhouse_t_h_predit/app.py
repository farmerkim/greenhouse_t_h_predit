import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import requests
import subprocess 

from google import genai 
from future_prediction import GreenhouseFuturePredictor

# --- [Gemini 및 Telegram 설정] ---
# 🔑 Gemini 설정
GEMINI_API_KEY = "" # 사용자 키 유지

# 📢 Telegram 설정
TELEGRAM_BOT_TOKEN = "8544768473:AAGlHKkR_r7-IjxoUqBrxcJd3aD6vRPSmvQ"
TELEGRAM_CHAT_ID = "7078646539"

# 온실 환경 및 임계치 설정 (수정됨)
CROP_NAME = "방울토마토"
TEMP_THRESHOLD_HIGH = 25.0
TEMP_THRESHOLD_LOW = 20.0  # 20도 이하로 떨어지면 알람
HUMIDITY_THRESHOLD_LOW = 50.0

# Gemini 클라이언트 초기화
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini_client = None

# --- [페이지 설정] ---
st.set_page_config(page_title="온실 미기후 예측 시스템", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2E7D32; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2E7D32; }
</style>
""", unsafe_allow_html=True)

if 'predictor' not in st.session_state: st.session_state.predictor = None
if 'predictions' not in st.session_state: st.session_state.predictions = None

# --- [함수 정의] ---
def get_gemini_advice(pred_temp, pred_humid, crop_name):
    if not gemini_client: return "API 키 확인 필요"
    prompt = f"당신은 {crop_name} 재배 컨설턴트입니다. 온실 온도가 {pred_temp:.1f}°C, 습도가 {pred_humid:.1f}%로 위험 수준이 예측되었습니다. 즉시 취해야 할 조치사항 3가지를 알려주세요."
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e: return f"Gemini 오류: {e}"

def send_telegram_alert(msg):
    if not TELEGRAM_BOT_TOKEN: return "텔레그램 설정 필요"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
        return "✅ 알람 발송 성공"
    except Exception as e: return f"❌ 발송 실패: {e}"

def check_and_get_alert_data(df):
    """
    [수정됨] 6시간 예측 데이터 중 '최저/최고' 값을 검사하여 알람 발생 여부 결정
    """
    if df is None or df.empty: return None
    
    # 전체 예측 기간 통계
    min_temp = df['Predicted_inner_temp'].min()
    max_temp = df['Predicted_inner_temp'].max()
    min_hum = df['Predicted_inner_hum'].min()
    
    # 알람 조건 확인
    reasons = []
    alert_temp = min_temp # 기본값
    
    if max_temp >= TEMP_THRESHOLD_HIGH:
        reasons.append(f"⚠️ 고온 경보 (최고 {max_temp:.1f}°C)")
        alert_temp = max_temp
    
    if min_temp <= TEMP_THRESHOLD_LOW:
        reasons.append(f"⚠️ 저온 경보 (최저 {min_temp:.1f}°C)")
        alert_temp = min_temp
        
    if min_hum <= HUMIDITY_THRESHOLD_LOW:
        reasons.append(f"⚠️ 저습 경보 (최저 {min_hum:.1f}%)")
    
    if reasons:
        # 위험이 감지된 시간대 찾기 (가장 먼저 위험해지는 시간)
        risk_row = df[
            (df['Predicted_inner_temp'] <= TEMP_THRESHOLD_LOW) | 
            (df['Predicted_inner_temp'] >= TEMP_THRESHOLD_HIGH) |
            (df['Predicted_inner_hum'] <= HUMIDITY_THRESHOLD_LOW)
        ].iloc[0]
        
        return {
            'temp': risk_row['Predicted_inner_temp'],
            'humid': risk_row['Predicted_inner_hum'],
            'time': risk_row['Date&Time'].strftime('%H:%M'),
            'reason': ", ".join(reasons)
        }
    return None

def load_predictor():
    try:
        p = GreenhouseFuturePredictor('output/best_model.pth', 'output/cache')
        p.load_model_and_scalers()
        return p
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

# --- [차트 함수] ---
def create_combined_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date&Time'], y=df['Predicted_inner_temp'], name='온도(°C)', line=dict(color='#FF6B6B', width=3)))
    fig.add_trace(go.Scatter(x=df['Date&Time'], y=df['Predicted_inner_hum'], name='습도(%)', yaxis='y2', line=dict(color='#95E1D3', width=3)))
    
    # 임계치 라인 추가 (시각적 확인용)
    fig.add_hline(y=TEMP_THRESHOLD_LOW, line_dash="dot", line_color="blue", annotation_text="저온 임계치")
    fig.add_hline(y=TEMP_THRESHOLD_HIGH, line_dash="dot", line_color="red", annotation_text="고온 임계치")
    
    fig.update_layout(
        title='온실 온습도 예측 (6시간)', xaxis_title='시간',
        yaxis=dict(title='온도'), yaxis2=dict(title='습도', overlaying='y', side='right'),
        hovermode='x unified', height=400, template='plotly_white'
    )
    return fig

# --- [메인 함수] ---
def main():
    st.markdown('<div class="main-header">🌱 온실 미기후 예측 시스템</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ 설정")
        forecast_path = st.text_input("기상청 예보 데이터", "input/weather_forecast.csv")
        preprocessed_path = st.text_input("전처리 데이터", "output/preprocessed_data.csv")
        hours_to_predict = st.slider("예측 시간", 1, 24, 6)
        predict_button = st.button("🚀 예측 실행", type="primary", use_container_width=True)

    # ------------------------------------------------------------------
    # 1. 예측 실행 로직
    # ------------------------------------------------------------------
    if predict_button:
        # A. 단기 예보 다운로드
        st.info("📡 기상청 예보 데이터 다운로드 중...")
        SCRIPT = 'short_term_forecast_download.py'
        
        if os.path.exists(SCRIPT):
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                res = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, check=False)
                
                if res.returncode == 0:
                    st.success("✅ 예보 데이터 업데이트 완료")
                else:
                    st.error("❌ 다운로드 스크립트 오류")
                    st.error(res.stderr)
            except Exception as e:
                st.error(f"실행 오류: {e}")
        else:
            st.warning(f"⚠️ {SCRIPT} 없음")

        # B. 파일 확인
        if not os.path.exists(forecast_path):
            st.error(f"❌ 파일 없음: {forecast_path}")
            return

        # C. 모델 예측
        with st.spinner("🔮 미래 환경 예측 중..."):
            if st.session_state.predictor is None:
                st.session_state.predictor = load_predictor()
            
            if st.session_state.predictor:
                try:
                    preds = st.session_state.predictor.predict(
                        forecast_path=forecast_path,
                        preprocessed_path=preprocessed_path,
                        target_date=None, 
                        hours_to_predict=hours_to_predict
                    )
                    st.session_state.predictions = preds
                    
                    if preds is not None and not preds.empty:
                        st.success(f"✅ 예측 완료! ({len(preds)}개 데이터)")
                    else:
                        st.error("❌ 예측 결과 없음")
                except Exception as e:
                    st.error(f"예측 오류: {e}")

        # D. 알람 로직 (수정됨)
        alert = check_and_get_alert_data(st.session_state.predictions)
        if alert:
            st.warning(f"🚨 **위험 감지!** {alert['time']}경 {alert['reason']}")
            
            with st.spinner("🤖 Gemini 조언 생성 중..."):
                advice = get_gemini_advice(alert['temp'], alert['humid'], CROP_NAME)
            
            msg = f"""
🚨 *{CROP_NAME} 긴급 알람* 🚨

🛑 *위험 감지*: {alert['reason']}
⏰ *발생 예상*: {alert['time']}
🌡️ *예측 온도*: {alert['temp']:.1f}°C
💧 *예측 습도*: {alert['humid']:.1f}%

🤖 *Gemini 조언*:
{advice}
"""
            st.markdown(f"**Telegram 메시지 미리보기:**\n```\n{msg}\n```")
            status = send_telegram_alert(msg)
            st.info(f"알람 전송 상태: {status}")
        else:
            st.success("✅ 향후 6시간 동안 위험 구간 없음")

    # ------------------------------------------------------------------
    # 2. 결과 시각화
    # ------------------------------------------------------------------
    df = st.session_state.predictions

    if df is not None and not df.empty:
        st.divider()
        st.subheader(f"📊 {hours_to_predict}시간 예측 결과")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("최저 온도", f"{df['Predicted_inner_temp'].min():.1f}°C", delta_color="inverse")
        c2.metric("최고 온도", f"{df['Predicted_inner_temp'].max():.1f}°C")
        c3.metric("평균 습도", f"{df['Predicted_inner_hum'].mean():.1f}%")

        st.plotly_chart(create_combined_chart(df), use_container_width=True)
        
        with st.expander("🔍 데이터 상세 보기"):
            st.dataframe(df)

if __name__ == "__main__":

    main()
