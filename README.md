# Video Classifier API

비디오 파일을 분석하여 게임 유형을 분류하는 딥러닝 API 서비스입니다.

## 📋 프로젝트 개요

이 프로젝트는 비디오 파일을 입력받아 LOL(League of Legends)과 TFT(Teamfight Tactics) 게임을 구분하는 분류 모델을 제공합니다. CNN 기반의 UILayoutCNN 모델을 사용하여 비디오의 UI 레이아웃을 분석하고 게임 유형을 분류합니다.

## 🚀 주요 기능

### 1. 비디오 분류 API (`/result`)

- 비디오 파일 업로드 및 실시간 분류
- LOL/TFT 게임 구분
- 신뢰도 점수 제공 (0.6 이하시 "unknown" 반환)
- 각 클래스별 확률 점수 제공

### 2. 데이터셋 준비 API (`/data-set`)

- 여러 비디오 파일을 동시에 업로드
- 초당 1프레임씩 추출하여 훈련 데이터셋 생성
- 처리 결과 요약 제공

### 3. 모델 학습 (`train.py`)

- UILayoutCNN 모델 훈련
- 데이터 증강 및 Early Stopping 적용
- 성능 모니터링 및 로깅

## 🏗️ 아키텍처

### 모델 구조 (UILayoutCNN)

- **기반 모델**: ResNet18 (ImageNet 사전 훈련)
- **입력 처리**: 이미지를 4개 영역으로 분할하여 각각에서 특징 추출
- **분류기**: 512×4 → 256 → num_classes (2개 클래스)
- **출력**: LOL/TFT 분류 확률

### 4개 영역 분할

1. **우측하단**: [240:360, 480:640] (120×160)
2. **상단 중앙**: [0:120, 200:440] (120×240)
3. **하단 중앙**: [240:360, 200:440] (120×240)
4. **좌측 중앙**: [120:240, 0:213] (120×213)

## 📁 프로젝트 구조

```
lolight-analyze.api/
├── main.py              # FastAPI 서버 (분류 API)
├── train.py             # 모델 훈련 스크립트
├── ui_layout_cnn.py     # UILayoutCNN 모델 정의
├── utils.py             # 비디오 처리 유틸리티
├── best_model.pth       # 최고 성능 모델 (자동 생성)
├── classify_model.pth   # 최종 모델 (자동 생성)
├── requirements.txt     # 의존성 패키지
├── Dockerfile          # Docker 설정
└── data/               # 데이터 디렉토리
    ├── train/          # 훈련 데이터
    │   ├── lol/        # LOL 게임 이미지들
    │   └── tft/        # TFT 게임 이미지들
    ├── validation/     # 검증 데이터
    └── temp/           # 임시 처리 파일들
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 훈련

```bash
python train.py
```

- 데이터 경로: `data/train`, `data/validation`
- 배치 크기: 32
- 학습 에포크: 10
- 학습률: 1e-3
- 이미지 크기: 360×640

### 3. API 서버 실행

```bash
python main.py
```

- 서버 주소: `http://localhost:8082`
- API 문서: `http://localhost:8082/docs`

## 📊 API 엔드포인트

### 1. 헬스 체크

```bash
GET /
```

서버 상태 확인

### 2. 비디오 분류

```bash
POST /result
```

**요청**: 비디오 파일 업로드
**응답**:

```json
{
  "label": "lol|tft|unknown",
  "score": 0.85,
  "lol_score": 0.85,
  "tft_score": 0.15
}
```

### 3. 데이터셋 준비

```bash
POST /data-set
```

**요청**: 여러 비디오 파일 + 디렉토리명
**응답**:

```json
{
  "total_frame_count": 1500,
  "processed_files": ["video1.mp4", "video2.mp4"],
  "total_files": 2,
  "successful_files": 2
}
```

## 🔧 설정

### 지원 파일 형식

- `.webm`, `.mp4`, `.avi`, `.mov`, `.mkv`

### 파일 크기 제한

- 최대 500MB

### 분류 임계값

- 신뢰도 점수 0.6 이하시 "unknown" 반환

## 📈 모델 성능

### 데이터 증강

- 좌우 반전 (50% 확률)
- ±15도 회전
- 색상 변화 (밝기, 대비, 채도, 색조)

### 정규화

- ImageNet 정규화 적용

### Early Stopping

- 5에포크 동안 성능 개선이 없으면 학습 중단

## 🐳 Docker 실행

```bash
# 이미지 빌드
docker build -t video-classifier .

# 컨테이너 실행
docker run -p 8082:8082 video-classifier
```

## 📝 로그

- 훈련 로그: `training.log`
- 실시간 콘솔 출력
- 배치별 진행상황 (5배치마다)
- 에포크별 성능 지표

## 🔍 모니터링

### 훈련 중 모니터링

- 에포크별 손실 및 정확도
- 배치별 진행상황
- 최고 성능 모델 자동 저장

### API 사용량 모니터링

- 파일 업로드 처리 상태
- 분류 결과 및 신뢰도 점수
- 오류 처리 및 로깅

## 🚨 주의사항

1. **GPU 사용**: CUDA가 설치된 환경에서 GPU 가속 사용 가능
2. **메모리 요구사항**: 대용량 비디오 처리시 충분한 RAM 필요
3. **임시 파일**: 업로드된 파일은 자동으로 정리됨
4. **모델 로딩**: 서버 시작시 모델이 메모리에 로드됨

## 📞 비고

- API 문서: `http://localhost:8082/docs` (Swagger UI)
- 로그 파일: `training.log`
- 오류 처리: HTTP 상태 코드 및 상세 메시지 제공
