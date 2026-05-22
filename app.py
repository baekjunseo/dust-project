import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from gtts import gTTS
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

API_KEY = '05689212741b54403fce3234b24229211ca799d76d07ddb84604d91c56f403bd'

st.set_page_config(page_title="초미세먼지 예측 대시보드", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"] { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; max-width: 100%; }
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.inner-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
}
.sec-title {
    font-size: 0.9rem;
    color: #8b949e;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #30363d;
}
.week-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 8px;
    text-align: center;
    margin: 4px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}
div[data-testid="stSelectbox"] > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] li {
    background: #161b22 !important;
    color: #e6edf3 !important;
}
div[data-baseweb="popover"] li:hover {
    background: #21262d !important;
}
p, span, div, h1, h2, h3, h4, label { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')


@st.cache_data
def load_data():
    DF = pd.read_csv('air_quality_clean.csv', encoding='UTF-8-SIG')
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# 실시간 API
@st.cache_data(ttl=3600)
def get_realtime(sido):
    try:
        url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
        params = {
            'serviceKey': API_KEY,
            'returnType': 'json',
            'numOfRows': '100',
            'pageNo': '1',
            'sidoName': sido,
            'ver': '1.0'
        }
        res = requests.get(url, params=params, timeout=5)
        items = res.json()['response']['body']['items']
        df_rt = pd.DataFrame(items)
        df_rt['pm25Value'] = pd.to_numeric(df_rt['pm25Value'], errors='coerce')
        df_rt['pm10Value'] = pd.to_numeric(df_rt['pm10Value'], errors='coerce')
        pm25 = df_rt['pm25Value'].mean()
        pm10 = df_rt['pm10Value'].mean()
        time = df_rt['dataTime'].iloc[0]
        return pm25, pm10, time
    except:
        return None, None, None

model = load_model()
df = load_data()

le_season = LabelEncoder()
le_region = LabelEncoder()
le_season.fit(df['season'])
le_region.fit(df['시도'])

def get_grade(v):
    if v <= 15:   return '좋음', '#2ea043'
    elif v <= 35: return '보통', '#d29922'
    elif v <= 75: return '나쁨', '#f0883e'
    else:         return '매우나쁨', '#f85149'

def get_season_str(m):
    if m in [3,4,5]: return '봄'
    elif m in [6,7,8]: return '여름'
    elif m in [9,10,11]: return '가을'
    else: return '겨울'

def predict(rdf, month, day, hour, weekday, pm25_now=None):
    lag1 = pm25_now if pm25_now else rdf['PM25'].iloc[-1]
    p = {
        'month': month, 'day': day, 'hour': hour, 'weekday': weekday,
        'season_enc': le_season.transform([get_season_str(month)])[0],
        'region_enc': le_region.transform([rdf['시도'].iloc[0]])[0],
        'PM10': rdf['PM10'].mean(), 'ws_avg': rdf['ws_avg'].mean(),
        'ta_avg': rdf['ta_avg'].mean(), 'hm_avg': rdf['hm_avg'].mean(),
        'pa_avg': rdf['pa_avg'].mean(), 'rn_day': rdf['rn_day'].mean(),
        'PM25_lag1': lag1,
        'PM25_lag3': rdf['PM25'].iloc[-3] if len(rdf)>=3 else lag1,
        'PM25_lag24': rdf['PM25'].iloc[-24] if len(rdf)>=24 else lag1,
        'PM25_avg6': rdf['PM25'].tail(6).mean(),
        'PM25_avg24': rdf['PM25'].tail(24).mean(),
    }
    return max(0, model.predict(pd.DataFrame([p]))[0])

now = datetime.now()
regions = sorted(df['시도'].unique())

# 헤더
hc1, hc2, hc3 = st.columns([3,1,1])
with hc1:
    st.markdown("""
    <div style='padding:20px 0 10px 0;'>
        <div style='font-size:1.6rem; font-weight:700; color:#e6edf3;'>
            초미세먼지 예측 대시보드
        </div>
    </div>""", unsafe_allow_html=True)
with hc2:
    selected = st.selectbox('지역 선택', regions,
        index=regions.index('서울') if '서울' in regions else 0)
with hc3:
    st.markdown(f"""
    <div style='text-align:right; padding:24px 0; color:#8b949e; font-size:0.82rem;'>
        {now.strftime('%Y.%m.%d %H:%M')} 기준
    </div>""", unsafe_allow_html=True)

st.markdown('<hr style="border-color:#30363d; margin:0 0 16px 0;">', unsafe_allow_html=True)

# 실시간 데이터
rt_pm25, rt_pm10, rt_time = get_realtime(selected)

rdf = df[df['시도']==selected].sort_values('datetime').tail(48)
latest = rdf.iloc[-1]

# 실시간 값 있으면 사용, 없으면 과거 데이터
current_pm25 = rt_pm25 if rt_pm25 else rdf['PM25'].iloc[-1]
current_pm10 = rt_pm10 if rt_pm10 else latest['PM10']

today_v = predict(rdf, now.month, now.day, now.hour, now.weekday(), current_pm25)
tmr = now + timedelta(days=1)
tmr_v = predict(rdf, tmr.month, tmr.day, 12, tmr.weekday(), current_pm25)

today_g, today_c = get_grade(today_v)
tmr_g, tmr_c = get_grade(tmr_v)
rt_g, rt_c = get_grade(current_pm25)
pm10_g, pm10_c = get_grade(current_pm10*0.6)

# 실시간 표시
if rt_pm25:
    st.markdown(f"""
    <div style="background:#1c2128; border:1px solid #30363d; border-radius:10px;
        padding:10px 16px; margin-bottom:12px; display:flex; gap:8px; align-items:center;">
        <div style="width:8px; height:8px; border-radius:50%; background:#2ea043;
            animation:pulse 2s infinite;"></div>
        <div style="font-size:0.82rem; color:#8b949e;">
            실시간 데이터 연동 중 &nbsp;|&nbsp; 측정시각: {rt_time}
        </div>
    </div>""", unsafe_allow_html=True)

# 음성
voice_txt = f"오늘 {selected} 초미세먼지는 {rt_g}입니다. 현재 농도 {current_pm25:.0f} 마이크로그램. 내일은 {tmr_v:.0f} 마이크로그램으로 {tmr_g} 예상됩니다."
st.markdown(f"""
<div style="background:#1c2128; border:1px solid #238636; border-radius:10px;
    padding:14px 18px; margin-bottom:16px; display:flex; gap:12px; align-items:flex-start;">
    <div style="background:#238636; border-radius:8px; width:36px; height:36px;
        display:flex; align-items:center; justify-content:center;
        flex-shrink:0; font-size:1.1rem; color:white; margin-top:2px;">&#9835;</div>
    <div style="flex:1; min-width:0;">
        <div style="font-size:0.78rem; color:#8b949e; margin-bottom:5px;">AI 음성 안내</div>
        <div style="font-size:0.9rem; color:#e6edf3; line-height:1.65; word-break:keep-all;">
            "{voice_txt}"
        </div>
    </div>
</div>""", unsafe_allow_html=True)

tts = gTTS(text=voice_txt, lang='ko')
tts.save('voice.mp3')
with open('voice.mp3','rb') as f:
    st.audio(f.read(), format='audio/mp3', autoplay=True)

# 지표 4개
c1,c2,c3,c4 = st.columns(4)
for col, lbl, val, unit, grade, color, icon in [
    (c1,'실시간 PM2.5', f'{current_pm25:.0f}','μg/m³', rt_g, rt_c, '&#127787;'),
    (c2,'내일 예측 PM2.5', f'{tmr_v:.0f}','μg/m³', tmr_g, tmr_c, '&#128197;'),
    (c3,'실시간 PM10', f'{current_pm10:.0f}','μg/m³', pm10_g, pm10_c, '&#127748;'),
    (c4,'모델 정확도','91','%','XGBoost','#2ea043','&#129302;'),
]:
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:4px;">{icon}</div>
            <div style="font-size:0.82rem; color:#8b949e;">{lbl}</div>
            <div style="font-size:2.5rem; font-weight:700; color:{color};
                margin:10px 0; line-height:1;">{val}</div>
            <div style="font-size:0.8rem; color:#8b949e; margin-bottom:8px;">{unit}</div>
            <span class="badge" style="background:{color}22; color:{color};
                border:1px solid {color}44;">{grade}</span>
        </div>""", unsafe_allow_html=True)

# 7일 예측 카드 (탭으로 PM2.5 / PM10 구분)
dates_7 = [now + timedelta(days=i) for i in range(7)]
pred_7 = [predict(rdf, d.month, d.day, 12, d.weekday(), current_pm25) for d in dates_7]
pred_7_pm10 = [v * 1.8 for v in pred_7]  # PM10 추정 (PM2.5 * 1.8 비율)
day_names = ['오늘','내일','모레','3일후','4일후','5일후','6일후']
weekday_kr = ['월','화','수','목','금','토','일']
weather_icons = ['&#9728;','&#9925;','&#9729;','&#127783;','&#127781;','&#9728;','&#9925;']

def get_grade_pm10(v):
    if v <= 30:   return '좋음', '#2ea043'
    elif v <= 80: return '보통', '#d29922'
    elif v <= 150: return '나쁨', '#f0883e'
    else:         return '매우나쁨', '#f85149'

tab1, tab2 = st.tabs(['&#127787; 초미세먼지 PM2.5', '&#127748; 미세먼지 PM10'])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">&#128197; {selected} 7일 PM2.5 예측</div>',
                unsafe_allow_html=True)
    cols_7 = st.columns(7)
    for i, (col, d, v, dn) in enumerate(zip(cols_7, dates_7, pred_7, day_names)):
        g, c = get_grade(v)
        wd = weekday_kr[d.weekday()]
        with col:
            st.markdown(f"""
            <div class="week-card" style="border-top:3px solid {c};">
                <div style="font-size:0.78rem; color:#8b949e;">{dn}</div>
                <div style="font-size:0.78rem; color:#8b949e; margin-bottom:6px;">
                    {d.strftime('%m/%d')} ({wd})
                </div>
                <div style="font-size:1.5rem; margin:6px 0;">{weather_icons[i]}</div>
                <div style="font-size:1.9rem; font-weight:700; color:{c}; line-height:1.2;">
                    {v:.0f}
                </div>
                <div style="font-size:0.72rem; color:#8b949e; margin:4px 0;">μg/m³</div>
                <span style="background:{c}22; color:{c}; border:1px solid {c}44;
                    padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:600;">
                    {g}
                </span>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">&#128197; {selected} 7일 PM10 예측</div>',
                unsafe_allow_html=True)
    cols_7b = st.columns(7)
    for i, (col, d, v, dn) in enumerate(zip(cols_7b, dates_7, pred_7_pm10, day_names)):
        g, c = get_grade_pm10(v)
        wd = weekday_kr[d.weekday()]
        with col:
            st.markdown(f"""
            <div class="week-card" style="border-top:3px solid {c};">
                <div style="font-size:0.78rem; color:#8b949e;">{dn}</div>
                <div style="font-size:0.78rem; color:#8b949e; margin-bottom:6px;">
                    {d.strftime('%m/%d')} ({wd})
                </div>
                <div style="font-size:1.5rem; margin:6px 0;">{weather_icons[i]}</div>
                <div style="font-size:1.9rem; font-weight:700; color:{c}; line-height:1.2;">
                    {v:.0f}
                </div>
                <div style="font-size:0.72rem; color:#8b949e; margin:4px 0;">μg/m³</div>
                <span style="background:{c}22; color:{c}; border:1px solid {c}44;
                    padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:600;">
                    {g}
                </span>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 막대그래프 + 행동가이드
gl, gr = st.columns([3,2])
with gl:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[d.strftime('%m/%d') for d in dates_7],
        y=pred_7,
        marker_color=[get_grade(v)[1] for v in pred_7],
        text=[f'{v:.0f}' for v in pred_7],
        textposition='outside',
        textfont=dict(color='#e6edf3', size=13),
        width=0.5,
    ))
    fig.add_hline(y=15, line_dash='dash', line_color='#2ea043', opacity=0.6,
                  annotation_text='좋음(15)', annotation_font_color='#2ea043',
                  annotation_position='right')
    fig.add_hline(y=35, line_dash='dash', line_color='#d29922', opacity=0.6,
                  annotation_text='보통(35)', annotation_font_color='#d29922',
                  annotation_position='right')
    fig.add_hline(y=75, line_dash='dash', line_color='#f85149', opacity=0.6,
                  annotation_text='나쁨(75)', annotation_font_color='#f85149',
                  annotation_position='right')
    fig.update_layout(
        height=300, plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        font=dict(color='#8b949e', size=11),
        showlegend=False,
        margin=dict(l=0,r=80,t=30,b=0),
        xaxis=dict(gridcolor='#30363d', color='#8b949e'),
        yaxis=dict(gridcolor='#30363d', color='#8b949e',
                   title='PM2.5 (μg/m³)', range=[0, max(pred_7)*1.4])
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">&#128202; 7일 PM2.5 예측 농도</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with gr:
    if rt_g in ['나쁨','매우나쁨']:
        guides = [
            ('#f85149','&#128683;','외출 자제 권장','불필요한 외출은 삼가세요'),
            ('#f0883e','&#128576;','KF94 마스크 착용','외출 시 반드시 착용하세요'),
            ('#58a6ff','&#127968;','실내 활동 권장','운동은 실내에서 하세요'),
            ('#8b949e','&#128168;','환기 자제','창문을 닫아두세요'),
        ]
    elif rt_g == '보통':
        guides = [
            ('#d29922','&#128084;','마스크 권장','민감군은 마스크를 착용하세요'),
            ('#58a6ff','&#127939;','야외활동 주의','장시간 야외활동 자제하세요'),
            ('#2ea043','&#128168;','환기 가능','잠깐씩 환기하세요'),
        ]
    else:
        guides = [
            ('#2ea043','&#127774;','외출하기 좋은 날','야외활동을 즐기세요'),
            ('#2ea043','&#128168;','환기 권장','창문을 열어 환기하세요'),
            ('#58a6ff','&#127939;','운동하기 좋은 날','야외 운동을 즐기세요'),
        ]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">&#9989; 행동 가이드</div>', unsafe_allow_html=True)
    for c, icon, t, d in guides:
        st.markdown(f"""
        <div class="inner-card" style="display:flex; gap:12px; align-items:center;">
            <div style="width:38px; height:38px; border-radius:10px;
                background:{c}22; display:flex; align-items:center;
                justify-content:center; flex-shrink:0; font-size:1.2rem;">{icon}</div>
            <div>
                <div style="font-size:0.88rem; font-weight:600;
                    color:#e6edf3; margin-bottom:2px;">{t}</div>
                <div style="font-size:0.8rem; color:#8b949e;">{d}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 원인분석 + 과거비교
al, ar = st.columns([3,2])
with al:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">&#128269; 원인 분석 (현재 기상 조건)</div>',
                unsafe_allow_html=True)
    factors = [
        ('&#128168; 풍속', latest['ws_avg']/10*100,
         f"{latest['ws_avg']:.1f} m/s", '낮을수록 미세먼지 나쁨'),
        ('&#128167; 습도', latest['hm_avg'],
         f"{latest['hm_avg']:.0f}%", '높을수록 오염물질 정체'),
        ('&#127783; 기압', (latest['pa_avg']-980)/60*100,
         f"{latest['pa_avg']:.0f} hPa", '높을수록 대기 정체'),
        ('&#127807; 전날 PM2.5', min(current_pm25/80*100,100),
         f"{current_pm25:.0f} μg/m³", '전날 농도 영향'),
        ('&#127968; 계절', 75 if now.month in [12,1,2,3] else 30,
         get_season_str(now.month), '겨울/봄 농도 높음'),
    ]
    for name, pct, display, hint in factors:
        pct = min(max(pct,0),100)
        c = '#f85149' if pct>70 else '#f0883e' if pct>40 else '#58a6ff'
        st.markdown(f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <div>
                    <span style="font-size:0.85rem; color:#e6edf3; font-weight:600;">
                        {name}
                    </span>
                    <span style="font-size:0.75rem; color:#8b949e; margin-left:8px;">
                        {hint}
                    </span>
                </div>
                <span style="font-size:0.85rem; color:{c}; font-weight:600;">{display}</span>
            </div>
            <div style="background:#0d1117; border-radius:6px; height:8px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{c}; border-radius:6px;">
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with ar:
    lm = df[(df['시도']==selected)&
            (df['datetime'].dt.month==(now.month-1 if now.month>1 else 12))]['PM25'].mean()
    ly = df[(df['시도']==selected)&
            (df['datetime'].dt.year==now.year-1)&
            (df['datetime'].dt.month==now.month)]['PM25'].mean()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">&#128197; 과거 대비 비교</div>',
                unsafe_allow_html=True)

    for lbl, val, diff_v in [
        ('현재 실시간', current_pm25, None),
        ('지난달 평균', lm, current_pm25-lm),
        ('작년 같은달', ly, current_pm25-ly),
    ]:
        st.markdown(f"""
        <div class="inner-card">
            <div style="font-size:0.8rem; color:#8b949e; margin-bottom:6px;">{lbl}</div>
            <div style="font-size:1.9rem; font-weight:700; color:{rt_c}; line-height:1.2;">
                {val:.0f}
                <span style="font-size:0.85rem; color:#8b949e;"> μg/m³</span>
            </div>
            {f'''<div style="font-size:0.82rem; color:{"#f85149" if diff_v>0 else "#2ea043"}; margin-top:5px;">
                {"▲" if diff_v>0 else "▼"} {abs(diff_v):.0f} μg/m³ {"높음" if diff_v>0 else "낮음"}
            </div>''' if diff_v is not None else ''}
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 등급 기준
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">&#128204; PM2.5 위험 등급 기준 (AirKorea 기준)</div>',
            unsafe_allow_html=True)
gc1,gc2,gc3,gc4 = st.columns(4)
for col, g, c, rng, desc, icon in [
    (gc1,'좋음','#2ea043','0 ~ 15 μg/m³','야외활동 자유롭게','&#128994;'),
    (gc2,'보통','#d29922','16 ~ 35 μg/m³','민감군 주의 필요','&#129000;'),
    (gc3,'나쁨','#f0883e','36 ~ 75 μg/m³','마스크 착용 권장','&#128993;'),
    (gc4,'매우나쁨','#f85149','76 μg/m³ 이상','외출 자제','&#128308;'),
]:
    with col:
        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid {c}44; border-radius:10px;
            padding:16px; text-align:center; border-top:3px solid {c};">
            <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
            <div style="font-size:1rem; font-weight:700; color:{c}; margin-bottom:6px;">{g}</div>
            <div style="font-size:0.85rem; color:#e6edf3; margin-bottom:4px;">{rng}</div>
            <div style="font-size:0.78rem; color:#8b949e;">{desc}</div>
        </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.8rem; padding:20px 0;">
    데이터: 에어코리아 실시간 API + 기상청 &nbsp;|&nbsp; AI 모델: XGBoost &nbsp;|&nbsp;
    2023~2025년 학습 &nbsp;|&nbsp; R² Score: 0.91
</div>""", unsafe_allow_html=True)
