# 🌫️ 초미세먼지 예측 대시보드

> 에어코리아 실시간 API + 기상 데이터 기반 XGBoost 머신러닝 모델을 활용한 초미세먼지(PM2.5) 예측 웹 대시보드

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://배포링크를입력하세요.streamlit.app)

---

## 📌 프로젝트 개요

전국 17개 시도의 실시간 미세먼지 데이터를 수집하고, 기상 조건을 결합하여 오늘/내일/7일치 PM2.5 농도를 예측하는 대시보드입니다.

- **기간:** 2026년 5월
- **데이터:** 2023 ~ 2025년 에어코리아 + 기상청 데이터
- **모델:** XGBoost (R² Score: 0.91)

---

## 🛠️ 사용 기술

| 분류 | 기술 |
|------|------|
| 언어 | Python |
| 데이터 수집 | 에어코리아 API, 기상청 API |
| 데이터 분석 | Pandas, NumPy |
| 머신러닝 | XGBoost, Scikit-learn |
| 시각화 | Plotly, Streamlit |
| 배포 | Streamlit Cloud, GitHub |
| 음성 | gTTS (Google Text-to-Speech) |

---

## 📊 주요 기능

- 🟢 **실시간 PM2.5 / PM10** — 에어코리아 API 실시간 연동
- 📅 **7일 예측** — XGBoost 모델 기반 초미세먼지 예측
- 📊 **예측 그래프** — 7일치 막대 그래프 시각화
- 🔍 **원인 분석** — 풍속, 습도, 기압 등 기상 요인 분석
- 📈 **과거 비교** — 지난달/작년 같은 달 평균과 비교
- ✅ **행동 가이드** — 등급별 맞춤 생활 가이드
- 🔊 **AI 음성 안내** — gTTS 기반 한국어 음성 브리핑

---

## 🔄 프로젝트 흐름

```
데이터 수집 → 전처리 → EDA → 피처 엔지니어링 → 모델 학습 → 배포
```

1. **데이터 수집** — 에어코리아 API, 기상청 API로 3년치 데이터 수집
2. **전처리** — 결측치 처리, 인코딩, 정규화
3. **EDA** — 상관관계, 계절별/지역별 분포 분석
4. **모델 학습** — XGBoost 모델 학습 및 하이퍼파라미터 튜닝
5. **배포** — Streamlit Cloud + GitHub 연동 자동 배포

---

## 📁 파일 구조

```
dust-project/
├── app.py              # Streamlit 메인 앱
├── final_data.csv      # 전처리된 최종 데이터
├── xgb_model.pkl       # 학습된 XGBoost 모델
└── requirements.txt    # 패키지 목록
```

---

## 🚀 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📉 모델 성능

| 지표 | 값 |
|------|----|
| R² Score | 0.91 |
| 알고리즘 | XGBoost |
| 학습 데이터 | 2023 ~ 2025년 (전국 17개 시도) |

---

## 📸 스크린샷


> 스크린<img width="2866" height="1198" alt="image" src="https://github.com/user-attachments/assets/cdabf346-a176-44e3-bc94-dc1cc22cc26f" />

샷을 추가하면 더욱 풍성한 포트폴리오가 됩니다!<img width="2878" height="1490" alt="image" src="https://github.com/user-attachments/assets/eb9cd7d2-39eb-4214-a2d4-0114259970ef" />

<img width="2852" height="1486" alt="image" src="https://github.com/user-attachments/assets/a7aa62c7-3a07-45c5-8ed3-5f4d9905e6ee" />


---

*데이터 출처: 에어코리아(한국환경공단), 기상청*
