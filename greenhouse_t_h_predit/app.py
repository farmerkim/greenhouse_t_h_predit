import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# future_prediction 모듈 import
from future_prediction import GreenhouseFuturePredictor

# 페이지 설정
st.set_page_config(
    page_title="온실 미기후 예측 시스템",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def load_predictor():
    """예측 모델 로드"""
    try:
        predictor = GreenhouseFuturePredictor(
            model_path='output/best_model.pth',
            scaler_dir='output/cache'
        )
        predictor.load_model_and_scalers()
        return predictor
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

def create_temperature_chart(df):
    """온도 예측 차트 생성"""
    fig = go.Figure()
    
    # 예측 온도
    fig.add_trace(go.Scatter(
        x=df['Date&Time'],
        y=df['Predicted_inner_temp'],
        mode='lines+markers',
        name='예측 온실 온도',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    # 외부 온도 (있으면)
    if 'outer_temp' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date&Time'],
            y=df['outer_temp'],
            mode='lines+markers',
            name='외부 온도',
            line=dict(color='#4ECDC4', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='온실 온도 예측',
        xaxis_title='시간',
        yaxis_title='온도 (°C)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_humidity_chart(df):
    """습도 예측 차트 생성"""
    fig = go.Figure()
    
    # 예측 습도
    fig.add_trace(go.Scatter(
        x=df['Date&Time'],
        y=df['Predicted_inner_hum'],
        mode='lines+markers',
        name='예측 온실 습도',
        line=dict(color='#95E1D3', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(149, 225, 211, 0.2)'
    ))
    
    # 외부 습도 (있으면)
    if 'outer_hum' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date&Time'],
            y=df['outer_hum'],
            mode='lines+markers',
            name='외부 습도',
            line=dict(color='#F38181', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='온실 습도 예측',
        xaxis_title='시간',
        yaxis_title='습도 (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_combined_chart(df):
    """온습도 통합 차트"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date&Time'],
        y=df['Predicted_inner_temp'],
        name='온도 (°C)',
        yaxis='y',
        line=dict(color='#FF6B6B', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date&Time'],
        y=df['Predicted_inner_hum'],
        name='습도 (%)',
        yaxis='y2',
        line=dict(color='#95E1D3', width=3)
    ))
    
    fig.update_layout(
        title='온실 온습도 통합 예측',
        xaxis_title='시간',
        yaxis=dict(
            title='온도 (°C)',
            titlefont=dict(color='#FF6B6B'),
            tickfont=dict(color='#FF6B6B')
        ),
        yaxis2=dict(
            title='습도 (%)',
            titlefont=dict(color='#95E1D3'),
            tickfont=dict(color='#95E1D3'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=450
    )
    
    return fig

def create_weather_comparison(df):
    """기상 조건 비교 차트"""
    if 'outer_temp' not in df.columns or 'outer_hum' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date&Time'],
        y=df['outer_temp'],
        name='외부 온도',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='외부 기상 조건',
        xaxis_title='시간',
        yaxis_title='온도 (°C)',
        template='plotly_white',
        height=300
    )
    
    return fig

def main():
    # 헤더
    st.markdown('<div class="main-header">🌱 온실 미기후 예측 시스템</div>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        st.markdown("---")
        
        # 파일 경로 설정
        st.subheader("📁 파일 경로")
        forecast_path = st.text_input(
            "기상청 예보 데이터",
            value="input/weather_forecast.csv"
        )
        preprocessed_path = st.text_input(
            "전처리 데이터",
            value="output/preprocessed_data.csv"
        )
        
        st.markdown("---")
        
        # 예측 설정
        st.subheader("🎯 예측 설정")
        
        # 날짜 선택
        use_specific_date = st.checkbox("특정 날짜 지정", value=False)
        target_date = None
        if use_specific_date:
            target_date = st.date_input("예측 날짜")
            target_date = target_date.strftime('%Y-%m-%d')
        
        hours_to_predict = st.slider(
            "예측 시간 (시간)",
            min_value=1,
            max_value=24,
            value=6,
            step=1
        )
        
        st.markdown("---")
        
        # 예측 실행 버튼
        predict_button = st.button("🚀 예측 실행", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # 정보
        st.markdown("""
        ### ℹ️ 사용 방법
        1. 파일 경로 확인
        2. 예측 설정 조정
        3. '예측 실행' 클릭
        
        ### 📊 모델 정보
        - 학습 기간: 4/16 ~ 10/27
        - 모델: LSTM
        - 예측 간격: 6시간
        """)
    
    # 메인 컨텐츠
    if predict_button:
        # 파일 존재 확인
        if not os.path.exists(forecast_path):
            st.error(f"❌ 예보 데이터를 찾을 수 없습니다: {forecast_path}")
            return
        
        if not os.path.exists(preprocessed_path):
            st.error(f"❌ 전처리 데이터를 찾을 수 없습니다: {preprocessed_path}")
            return
        
        # 예측 수행
        with st.spinner("🔄 모델 로딩 중..."):
            if st.session_state.predictor is None:
                st.session_state.predictor = load_predictor()
            
            if st.session_state.predictor is None:
                st.error("모델 로드에 실패했습니다.")
                return
        
        with st.spinner("🔮 예측 수행 중..."):
            try:
                predictions = st.session_state.predictor.predict(
                    forecast_path=forecast_path,
                    preprocessed_path=preprocessed_path,
                    target_date=target_date,
                    hours_to_predict=hours_to_predict
                )
                
                if predictions is not None:
                    st.session_state.predictions = predictions
                    st.success("✅ 예측이 완료되었습니다!")
                else:
                    st.error("예측에 실패했습니다.")
                    return
            
            except Exception as e:
                st.error(f"오류 발생: {e}")
                return
    
    # 예측 결과 표시
    if st.session_state.predictions is not None:
        df = st.session_state.predictions
        
        # 요약 통계
        st.markdown("## 📊 예측 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "평균 온도",
                f"{df['Predicted_inner_temp'].mean():.1f}°C",
                f"{df['Predicted_inner_temp'].max() - df['Predicted_inner_temp'].min():.1f}°C 변화"
            )
        
        with col2:
            st.metric(
                "평균 습도",
                f"{df['Predicted_inner_hum'].mean():.1f}%",
                f"{df['Predicted_inner_hum'].max() - df['Predicted_inner_hum'].min():.1f}% 변화"
            )
        
        with col3:
            st.metric(
                "최고 온도",
                f"{df['Predicted_inner_temp'].max():.1f}°C",
                f"at {df.loc[df['Predicted_inner_temp'].idxmax(), 'Date&Time'].strftime('%H:%M')}"
            )
        
        with col4:
            st.metric(
                "최저 온도",
                f"{df['Predicted_inner_temp'].min():.1f}°C",
                f"at {df.loc[df['Predicted_inner_temp'].idxmin(), 'Date&Time'].strftime('%H:%M')}"
            )
        
        st.markdown("---")
        
        # 차트
        st.markdown("## 📈 예측 결과 시각화")
        
        tab1, tab2, tab3, tab4 = st.tabs(["통합 뷰", "온도 상세", "습도 상세", "데이터 테이블"])
        
        with tab1:
            st.plotly_chart(create_combined_chart(df), use_container_width=True)
            
            # 기상 조건 비교
            weather_chart = create_weather_comparison(df)
            if weather_chart:
                st.plotly_chart(weather_chart, use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_temperature_chart(df), use_container_width=True)
            
            # 온도 분석
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **온도 분석**
                - 최고: {df['Predicted_inner_temp'].max():.1f}°C
                - 최저: {df['Predicted_inner_temp'].min():.1f}°C
                - 평균: {df['Predicted_inner_temp'].mean():.1f}°C
                - 표준편차: {df['Predicted_inner_temp'].std():.2f}°C
                """)
            
            with col2:
                # 온도 분포
                fig = px.box(df, y='Predicted_inner_temp', title='온도 분포')
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_humidity_chart(df), use_container_width=True)
            
            # 습도 분석
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **습도 분석**
                - 최고: {df['Predicted_inner_hum'].max():.1f}%
                - 최저: {df['Predicted_inner_hum'].min():.1f}%
                - 평균: {df['Predicted_inner_hum'].mean():.1f}%
                - 표준편차: {df['Predicted_inner_hum'].std():.2f}%
                """)
            
            with col2:
                # 습도 분포
                fig = px.box(df, y='Predicted_inner_hum', title='습도 분포')
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # 데이터 테이블
            st.dataframe(
                df.style.format({
                    'Predicted_inner_temp': '{:.1f}°C',
                    'Predicted_inner_hum': '{:.1f}%',
                    'outer_temp': '{:.1f}°C',
                    'outer_hum': '{:.1f}%',
                    'wind_speed': '{:.1f}m/s',
                    'rainfall': '{:.1f}mm'
                }),
                use_container_width=True
            )
            
            # CSV 다운로드
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # 시간별 상세 정보
        with st.expander("🕐 시간별 상세 정보"):
            for idx, row in df.iterrows():
                col1, col2, col3 = st.columns([2, 3, 3])
                
                with col1:
                    st.markdown(f"**{row['Date&Time'].strftime('%Y-%m-%d %H:%M')}**")
                    if 'Hours_Ahead' in row:
                        st.caption(f"{row['Hours_Ahead']}시간 후")
                
                with col2:
                    st.markdown(f"🌡️ **{row['Predicted_inner_temp']:.1f}°C** | "
                              f"💧 **{row['Predicted_inner_hum']:.1f}%**")
                
                with col3:
                    if 'outer_temp' in row and not pd.isna(row['outer_temp']):
                        st.markdown(f"🌤️ 외부: {row['outer_temp']:.1f}°C, "
                                  f"{row.get('outer_hum', 'N/A')}%")
                
                st.markdown("---")
    
    else:
        # 초기 화면
        st.info("""
        ### 👋 환영합니다!
        
        온실 미기후 예측 시스템은 기상청 단기예보 데이터를 활용하여 
        온실 내부의 온도와 습도를 예측합니다.
        
        **시작하려면:**
        1. 왼쪽 사이드바에서 파일 경로를 확인하세요
        2. 예측 설정을 조정하세요
        3. '예측 실행' 버튼을 클릭하세요
        
        """)
        
        # 샘플 이미지나 설명 추가 가능
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 정확한 예측
            LSTM 딥러닝 모델로 
            정확한 예측을 제공합니다
            """)
        
        with col2:
            st.markdown("""
            #### 📊 직관적 시각화
            다양한 차트로 
            결과를 쉽게 확인하세요
            """)
        
        with col3:
            st.markdown("""
            #### ⚡ 실시간 분석
            빠른 예측과 
            즉각적인 결과 제공
            """)

if __name__ == "__main__":
    main()