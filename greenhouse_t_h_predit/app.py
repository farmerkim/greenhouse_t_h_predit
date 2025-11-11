# app.py - LSTM 온실 예측 Streamlit 대시보드
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
from twilio.rest import Client
import os
from PIL import Image

# Twilio 설정
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
ALERT_PHONE_NUMBER = os.getenv('ALERT_PHONE_NUMBER', '')

class TwilioAlertSystem:
    def __init__(self):
        try:
            if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
                self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                self.enabled = True
            else:
                self.enabled = False
        except:
            self.enabled = False
    
    def send_alert(self, message):
        """SMS 알림 전송"""
        if not self.enabled:
            return False
        
        try:
            self.client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=ALERT_PHONE_NUMBER
            )
            return True
        except Exception as e:
            st.error(f"알림 전송 실패: {e}")
            return False
    
    def check_thresholds(self, temp, humidity, temp_min, temp_max, humid_min, humid_max):
        """임계값 체크 및 알림"""
        alerts = []
        
        if temp < temp_min:
            alerts.append(f"🥶 온도 경고: {temp:.1f}°C (최소: {temp_min}°C)")
        elif temp > temp_max:
            alerts.append(f"🔥 온도 경고: {temp:.1f}°C (최대: {temp_max}°C)")
        
        if humidity < humid_min:
            alerts.append(f"💧 습도 경고: {humidity:.1f}% (최소: {humid_min}%)")
        elif humidity > humid_max:
            alerts.append(f"💦 습도 경고: {humidity:.1f}% (최대: {humid_max}%)")
        
        if alerts:
            message = f"🌱 온실 환경 경고!\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(alerts)
            if self.send_alert(message):
                return alerts
        
        return alerts

@st.cache_data
def load_future_predictions(future_path='output/prediction_20251027_future.csv'):
    """미래 예측 결과를 로드 (10월 27일 데이터)"""
    try:
        df_future = pd.read_csv(future_path, parse_dates=['Date&Time'])
        return df_future
    except FileNotFoundError:
        return None

@st.cache_data
def load_predictions():
    """예측 결과 로드"""
    try:
        with open('models/predictions.pkl', 'rb') as f:
            predictions = pickle.load(f)
        return predictions
    except FileNotFoundError:
        return None


def create_forecast_plot(predictions):
    """예측 결과 시각화"""
    y_test = predictions['y_test']
    y_pred = predictions['y_pred']
    timestamps = pd.to_datetime(predictions['test_timestamps'])
    target_columns = predictions['target_columns']
    
    # 2개 서브플롯 (온도, 습도)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{target_columns[0]} 예측', f'{target_columns[1]} 예측'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # 온도 그래프
    fig.add_trace(
        go.Scatter(
            x=timestamps, 
            y=y_test[:, 0],
            mode='lines',
            name='실제 온도',
            line=dict(color='#2E86DE', width=2),
            legendgroup='temp'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred[:, 0],
            mode='lines',
            name='예측 온도',
            line=dict(color='#EE5A6F', width=2, dash='dash'),
            legendgroup='temp'
        ),
        row=1, col=1
    )
    
    # 습도 그래프
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_test[:, 1],
            mode='lines',
            name='실제 습도',
            line=dict(color='#10AC84', width=2),
            legendgroup='humid',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred[:, 1],
            mode='lines',
            name='예측 습도',
            line=dict(color='#F79F1F', width=2, dash='dash'),
            legendgroup='humid',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="시간", row=2, col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="습도 (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_scatter_plot(predictions):
    """실제값 vs 예측값 산점도"""
    y_test = predictions['y_test']
    y_pred = predictions['y_pred']
    target_columns = predictions['target_columns']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{target_columns[0]}: 실제 vs 예측', f'{target_columns[1]}: 실제 vs 예측')
    )
    
    # 온도
    fig.add_trace(
        go.Scatter(
            x=y_test[:, 0],
            y=y_pred[:, 0],
            mode='markers',
            name='온도',
            marker=dict(
                color='#2E86DE',
                size=8,
                opacity=0.6,
                line=dict(width=1, color='white')
            )
        ),
        row=1, col=1
    )
    
    # 완벽한 예측 선 (온도)
    min_temp, max_temp = y_test[:, 0].min(), y_test[:, 0].max()
    fig.add_trace(
        go.Scatter(
            x=[min_temp, max_temp],
            y=[min_temp, max_temp],
            mode='lines',
            name='완벽한 예측',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 습도
    fig.add_trace(
        go.Scatter(
            x=y_test[:, 1],
            y=y_pred[:, 1],
            mode='markers',
            name='습도',
            marker=dict(
                color='#10AC84',
                size=8,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 완벽한 예측 선 (습도)
    min_humid, max_humid = y_test[:, 1].min(), y_test[:, 1].max()
    fig.add_trace(
        go.Scatter(
            x=[min_humid, max_humid],
            y=[min_humid, max_humid],
            mode='lines',
            name='완벽한 예측',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="실제값", row=1, col=1)
    fig.update_yaxes(title_text="예측값", row=1, col=1)
    fig.update_xaxes(title_text="실제값", row=1, col=2)
    fig.update_yaxes(title_text="예측값", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    
    return fig


def create_error_analysis(predictions):
    """오차 분석"""
    y_test = predictions['y_test']
    y_pred = predictions['y_pred']
    target_columns = predictions['target_columns']
    
    temp_error = y_test[:, 0] - y_pred[:, 0]
    humid_error = y_test[:, 1] - y_pred[:, 1]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{target_columns[0]} 오차 분포', f'{target_columns[1]} 오차 분포')
    )
    
    fig.add_trace(
        go.Histogram(
            x=temp_error,
            nbinsx=30,
            name='온도 오차',
            marker_color='#5F27CD',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=humid_error,
            nbinsx=30,
            name='습도 오차',
            marker_color='#00D2D3',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="오차 (실제 - 예측)", row=1, col=1)
    fig.update_xaxes(title_text="오차 (실제 - 예측)", row=1, col=2)
    fig.update_yaxes(title_text="빈도", row=1, col=1)
    fig.update_yaxes(title_text="빈도", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    
    return fig


def main():
    st.set_page_config(
        page_title="온실 미기후 LSTM 예측 시스템",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌱 온실 미기후 LSTM 예측 시스템")
        st.markdown("**딥러닝 기반 온도 및 습도 예측**")
    
    st.markdown("---")
    
    # 예측 데이터 로드
    predictions = load_predictions()
    df_future = load_future_predictions()

    if predictions is None:
        st.error("❌ 예측 결과를 찾을 수 없습니다. `python model_training.py`를 먼저 실행하세요!")
        st.info("""
        **실행 순서:**
        1. `python asos_download.py` - 기상 데이터 다운로드
        2. `python data_preprocessing.py` - 데이터 전처리
        3. `python model_training.py` - LSTM 모델 학습
        4. `streamlit run app.py` - 대시보드 실행
        """)
        return
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        st.markdown("### 📊 모델 정보")
        st.info(f"""
        **모델:** LSTM (Long Short-Term Memory)  
        **예측 대상:** {', '.join(predictions['target_columns'])}  
        **테스트 기간:** {len(predictions['y_test'])}시간
        """)
        
        st.markdown("---")
        st.markdown("### 🔔 알림 임계값 설정")
        
        temp_min = st.number_input("최저 온도 (°C)", value=15.0, step=0.5, key='temp_min')
        temp_max = st.number_input("최고 온도 (°C)", value=35.0, step=0.5, key='temp_max')
        humid_min = st.number_input("최저 습도 (%)", value=40.0, step=1.0, key='humid_min')
        humid_max = st.number_input("최고 습도 (%)", value=80.0, step=1.0, key='humid_max')
        
        enable_alerts = st.checkbox("실시간 알림 활성화", value=False)
        
        if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
            st.warning("⚠️ Twilio 설정이 필요합니다 (.env 파일)")
    
    # 메트릭 표시
    metrics = predictions['metrics']
    target_columns = predictions['target_columns']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{target_columns[0]} MAE",
            f"{metrics[target_columns[0]]['mae']:.3f}",
            help="평균 절대 오차"
        )
    with col2:
        st.metric(
            f"{target_columns[0]} R²",
            f"{metrics[target_columns[0]]['r2']:.3f}",
            help="결정 계수 (1에 가까울수록 좋음)"
        )
    with col3:
        st.metric(
            f"{target_columns[1]} MAE",
            f"{metrics[target_columns[1]]['mae']:.3f}",
            help="평균 절대 오차"
        )
    with col4:
        st.metric(
            f"{target_columns[1]} R²",
            f"{metrics[target_columns[1]]['r2']:.3f}",
            help="결정 계수 (1에 가까울수록 좋음)"
        )
    
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([ 
        "📊 모델 성능 (9/26~10/26)",
        "🚀 10월 27일 예측", # 새로운 탭
        "🎯 오차 분석",
        "🔔 실시간 모니터링",
        "📋 상세 데이터"
    ])
    
    with tab1:
        st.subheader("모델 성능 검증 결과 (9/26 ~ 10/26)")
        fig = create_forecast_plot(predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"""
            **{target_columns[0]} 성능**  
            - MAE: {metrics[target_columns[0]]['mae']:.3f}  
            - RMSE: {metrics[target_columns[0]]['rmse']:.3f}  
            - R²: {metrics[target_columns[0]]['r2']:.3f}
            """)
        with col2:
            st.success(f"""
            **{target_columns[1]} 성능**  
            - MAE: {metrics[target_columns[1]]['mae']:.3f}  
            - RMSE: {metrics[target_columns[1]]['rmse']:.3f}  
            - R²: {metrics[target_columns[1]]['r2']:.3f}
            """)
    
    with tab2:
        st.subheader("10월 27일 최종 예측 결과")
        if df_future is not None:
            # 미래 예측 시각화 로직 (실제값은 없으므로 예측값만 표시)
            fig_future = make_subplots(rows=2, cols=1, subplot_titles=('10월 27일 온도 예측', '10월 27일 습도 예측'))
            
            # 온도
            fig_future.add_trace(go.Scatter(x=df_future['Date&Time'], y=df_future['Predicted_inner_temp'], mode='lines', name='예측 온도'), row=1, col=1)
            # 습도
            fig_future.add_trace(go.Scatter(x=df_future['Date&Time'], y=df_future['Predicted_inner_hum'], mode='lines', name='예측 습도'), row=2, col=1)

            fig_future.update_xaxes(title_text="시간", row=2, col=1)
            fig_future.update_yaxes(title_text="온도 (°C)", row=1, col=1)
            fig_future.update_yaxes(title_text="습도 (%)", row=2, col=1)
            fig_future.update_layout(height=700, showlegend=True, hovermode='x unified', template='plotly_white')

            st.plotly_chart(fig_future, use_container_width=True)
            st.dataframe(df_future, use_container_width=True)
        else:
            st.warning("⚠️ 10월 27일 예측 결과를 찾을 수 없습니다. `python future_prediction.py`를 실행하세요.")
    
    with tab3:
        st.subheader("예측 오차 분석")
        
        # 최근 예측값의 시간대를 명시
        latest_timestamp = pd.to_datetime(predictions['test_timestamps'][-1]).strftime('%Y년 %m월 %d일 %H시')
        st.info(f"✨ **모니터링 기준 시점:** {latest_timestamp} 예측값")

        y_pred = predictions['y_pred']
        latest_temp = y_pred[-1, 0]
        latest_humid = y_pred[-1, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 온도 게이지
            delta_temp = "정상" if temp_min <= latest_temp <= temp_max else "⚠️ 범위 이탈"
            st.metric(
                "현재 예측 온도",
                f"{latest_temp:.1f}°C",
                delta=delta_temp
            )
            
            if latest_temp < temp_min:
                st.error(f"🥶 온도가 최저 기준({temp_min}°C)보다 낮습니다!")
            elif latest_temp > temp_max:
                st.error(f"🔥 온도가 최고 기준({temp_max}°C)보다 높습니다!")
            else:
                st.success("✅ 온도가 정상 범위 내에 있습니다")
        
        with col2:
            # 습도 게이지
            delta_humid = "정상" if humid_min <= latest_humid <= humid_max else "⚠️ 범위 이탈"
            st.metric(
                "현재 예측 습도",
                f"{latest_humid:.1f}%",
                delta=delta_humid
            )
            
            if latest_humid < humid_min:
                st.error(f"💧 습도가 최저 기준({humid_min}%)보다 낮습니다!")
            elif latest_humid > humid_max:
                st.error(f"💦 습도가 최고 기준({humid_max}%)보다 높습니다!")
            else:
                st.success("✅ 습도가 정상 범위 내에 있습니다")
        
        st.markdown("---")
        
        # 알림 테스트
        alert_system = TwilioAlertSystem()
        
        if enable_alerts and alert_system.enabled:
            if st.button("🔔 알림 테스트", type="primary"):
                with st.spinner("알림 전송 중..."):
                    alerts = alert_system.check_thresholds(
                        latest_temp, latest_humid,
                        temp_min, temp_max, humid_min, humid_max
                    )
                    
                    if alerts:
                        for alert in alerts:
                            st.warning(alert)
                        st.success("✅ SMS 알림이 전송되었습니다!")
                    else:
                        st.info("✅ 모든 값이 정상 범위 내에 있습니다.")
        elif enable_alerts and not alert_system.enabled:
            st.warning("⚠️ Twilio가 설정되지 않아 알림을 보낼 수 없습니다.")
    
    with tab4:
        st.subheader("예측 결과 상세 데이터")
        
        # 데이터프레임 생성
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        timestamps = pd.to_datetime(predictions['test_timestamps'])
        
        results_df = pd.DataFrame({
            '시간': timestamps,
            f'실제_{target_columns[0]}': y_test[:, 0],
            f'예측_{target_columns[0]}': y_pred[:, 0],
            f'{target_columns[0]}_오차': y_test[:, 0] - y_pred[:, 0],
            f'실제_{target_columns[1]}': y_test[:, 1],
            f'예측_{target_columns[1]}': y_pred[:, 1],
            f'{target_columns[1]}_오차': y_test[:, 1] - y_pred[:, 1]
        })
        
        st.dataframe(results_df, use_container_width=True, height=400)
        
        # CSV 다운로드
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 예측 결과 다운로드 (CSV)",
            data=csv,
            file_name=f'lstm_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            type="primary"
        )
        
        # 통계 요약
        st.markdown("### 📊 통계 요약")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **{target_columns[0]} 통계**  
            - 실제 평균: {y_test[:, 0].mean():.2f}  
            - 예측 평균: {y_pred[:, 0].mean():.2f}  
            - 실제 표준편차: {y_test[:, 0].std():.2f}  
            - 예측 표준편차: {y_pred[:, 0].std():.2f}
            """)
        
        with col2:
            st.info(f"""
            **{target_columns[1]} 통계**  
            - 실제 평균: {y_test[:, 1].mean():.2f}  
            - 예측 평균: {y_pred[:, 1].mean():.2f}  
            - 실제 표준편차: {y_test[:, 1].std():.2f}  
            - 예측 표준편차: {y_pred[:, 1].std():.2f}
            """)


if __name__ == "__main__":
    main()