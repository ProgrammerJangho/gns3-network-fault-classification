# -*- coding: utf-8 -*-
"""
config.py
-------------------------------------------------------------------
프로젝트 전역 설정 한 곳에 모음.
- 장애 클래스 / GNS3 링크 필터 파라미터
- 합성 데이터 생성 프로파일(synth_data.py)
- 피처 정의 / 유효 범위(augment.py 의 clip 기준)
- GNS3 접속 정보 (v2.2 / v3 둘 다 지원)
- 파일 경로

여기 값만 고치면 collect / augment / train / demo 가 전부 따라감.
"""

import os

# =====================================================================
# 1. 장애 클래스
# =====================================================================
# 코드 내부에서는 ASCII 키를 쓰고(그래프 폰트 깨짐 방지),
# 화면 출력용 한글 이름은 KOR_NAMES 로 매핑.
CLASSES = ["normal", "high_delay", "packet_loss"]

KOR_NAMES = {
    "normal":      "정상",
    "high_delay":  "높은지연",
    "packet_loss": "패킷손실",
}

# =====================================================================
# 2. 피처
# =====================================================================
# 에피소드(=ping 100회) 당 1개의 피처 벡터.
FEATURES = [
    "loss_pct",   # 손실률 (%)
    "rtt_mean",   # 평균 RTT (ms)
    "rtt_std",    # RTT 표준편차 (ms)
    "rtt_min",    # 최소 RTT (ms)
    "rtt_max",    # 최대 RTT (ms)
    "rtt_range",  # rtt_max - rtt_min (ms)
    "jitter",     # 연속 RTT 차의 평균 절대값 (ms)
]

# 100% 손실(링크다운)일 때 RTT 가 없으므로 이 천장값으로 채움.
TIMEOUT_CEILING_MS = 2000.0

# 에피소드당 ping 횟수.  링크다운은 전 패킷 타임아웃이라 이 값이 클수록 느림.
# dynamips+uBridge 환경에선 100회는 느리고 버퍼 압박으로 손실이 생겨 30으로 낮춤
# (피처 평균/표준편차/지터는 30회로도 충분히 안정적).
PING_COUNT = 30

# =====================================================================
# 3. GNS3 링크 필터 (collect.py 가 P1-P2 링크에 주입)
#    PUT /v{2,3}/projects/{pid}/links/{lid}  body={"filters": <아래>}
# =====================================================================
# 참고: link_down(전면 단절)과 congestion(지연+손실 복합)은 제외함.
#  - link_down: loss==100% 임계값만으로 자명, 100% 손실이 OSPF Hello 막아 코어 다운
#  - congestion: delay+packet_loss 복합이 dynamips 큐를 흔들어 OSPF dead-timer 만료 → 측정 불안정
#  단일 요인 장애 3종(정상/높은지연/패킷손실)에 집중. "정상 vs 장애"는 룰, "장애 종류"는 ML.
FILTERS = {
    "normal":      {},                        # 필터 없음
    "high_delay":  {"delay": [80, 10]},        # 지연80ms, 지터10ms
    "packet_loss": {"packet_loss": [15]},      # 손실 15%
}

# 샘플마다 필터 수치를 살짝 흔들어 다양화 (±비율).  frequency_drop 은 흔들지 않음.
FILTER_JITTER_FRAC = 0.15

# =====================================================================
# 4. 합성 데이터 프로파일 (synth_data.py — GNS3 없이 파이프라인 검증용)
#    실제 GNS3 동작을 모사해 패킷별 RTT 를 만들고, 동일 feature 함수로 벡터화.
#      extra_delay : 정상 대비 추가 지연(ms)        -> rtt_mean 상승
#      jitter_std  : 패킷별 RTT 가우시안 노이즈 std -> rtt_std / jitter 상승
#      loss        : 손실 확률(0~1)                 -> loss_pct
# =====================================================================
BASE_RTT_MS = 10.0   # PE1->...->PE2 4홉 정상 왕복 기준선(ms)

SYNTH_PROFILES = {
    "normal":      {"extra_delay": 0,  "jitter_std": 1.5, "loss": 0.00},
    "high_delay":  {"extra_delay": 80, "jitter_std": 6.0, "loss": 0.01},  # 큰 지연, 중간 지터
    "packet_loss": {"extra_delay": 0,  "jitter_std": 2.5, "loss": 0.15},  # 정상RTT + 높은 손실
}

# 합성 시 샘플마다 프로파일을 흔드는 비율(다양화).
SYNTH_JITTER_FRAC = 0.15

# =====================================================================
# 5. 피처 유효 범위 (augment.py clip 기준)
# =====================================================================
FEATURE_BOUNDS = {
    "loss_pct":  (0.0, 100.0),
    "rtt_mean":  (0.0, TIMEOUT_CEILING_MS),
    "rtt_std":   (0.0, TIMEOUT_CEILING_MS),
    "rtt_min":   (0.0, TIMEOUT_CEILING_MS),
    "rtt_max":   (0.0, TIMEOUT_CEILING_MS),
    "rtt_range": (0.0, TIMEOUT_CEILING_MS),
    "jitter":    (0.0, TIMEOUT_CEILING_MS),
}

# =====================================================================
# 6. 수집 / 증강 개수
# =====================================================================
SAMPLES_PER_CLASS = 30    # 클래스별 실측(또는 합성) 원본 개수 M
N_PER_CLASS       = 200   # 증강 후 클래스별 총 개수 N (원본 포함)
AUG_NOISE_SCALE   = 1.0   # 증강 노이즈 = 관측 std * 이 값

# =====================================================================
# 7. GNS3 접속 (collect.py / demo.py)
# =====================================================================
# --- API 버전: "v2"(2.2 기준) 또는 "v3"(3.0+, JWT 인증) ---
GNS3_API_VERSION = "v2"

GNS3_HOST = "127.0.0.1"
GNS3_PORT = 3080
GNS3_SCHEME = "http"          # v3 에서 TLS 켰으면 "https"

# v2(2.2)에서 "Protect server with password" 켰으면 True 로.
# v3 는 항상 인증이므로 이 값과 무관하게 로그인함.
GNS3_AUTH = True

# 인증 정보 (v2 Basic Auth / v3 로그인 공용)
# 2.2: %APPDATA%\GNS3\gns3_server.conf 의 [Server] user/password 값
# 깃허브에 비밀번호가 올라가지 않도록 환경변수에서 읽음.
#   Windows(cmd):  set GNS3_PASSWORD=서버비밀번호
#   PowerShell  :  $env:GNS3_PASSWORD="서버비밀번호"
GNS3_USERNAME = os.environ.get("GNS3_USERNAME", "admin")
GNS3_PASSWORD = os.environ.get("GNS3_PASSWORD", "")

# 프로젝트: 이름으로 자동 검색하거나 ID 직접 지정(둘 중 하나).
GNS3_PROJECT_NAME = "AI-project"   # GNS3 프로젝트 이름
GNS3_PROJECT_ID   = None         # 알면 직접: "xxxxxxxx-xxxx-..."

# 장애를 주입할 코어 링크의 양 끝 노드 이름 (이 둘을 잇는 링크를 자동으로 찾음).
# dynamips 직결 링크는 uBridge가 불안정해 필터가 안 먹어서, P1-P2 사이에
# Ethernet switch(Switch1)를 삽입함. 스위치 링크엔 uBridge가 안정적으로 붙음.
CORE_LINK_ENDPOINTS = ("P1", "Switch1")

# 측정 경로: SRC_VPCS 콘솔에 텔넷 접속 -> DST_VPCS_IP 로 ping.
SRC_VPCS_NAME = "PC1"         # PE1 쪽 (IP 10.0.1.2/24, GW 10.0.1.1)
DST_VPCS_IP   = "10.0.2.2"    # VPCS2(PE2 쪽) 에 설정한 IP (10.0.2.2/24, GW 10.0.2.1)
CONSOLE_HOST  = "127.0.0.1"   # 콘솔 텔넷 호스트 (보통 GNS3_HOST 와 동일)
VPCS_PROMPT   = "VPCS>"       # VPCS 기본 프롬프트

# 타이밍
FILTER_SETTLE_S       = 5.0    # 필터 주입 후 안정화 대기(uBridge/dynamips 안정 시간)
CONSOLE_READ_MAX_WAIT = max(30.0, PING_COUNT * 1.3)  # ping 출력 최대 대기(s)
CONSOLE_IDLE_BREAK_S  = 1.2    # 이 시간 동안 신규 출력 없으면 종료로 간주

# =====================================================================
# 8. 경로
# =====================================================================
RAW_CSV     = "dataset_raw.csv"   # 실측/합성 원본 (collect.py 또는 synth_data.py)
DATASET_CSV = "dataset.csv"       # 증강 완료본 (augment.py -> train/demo 가 사용)
MODEL_DIR   = "models"
PLOTS_DIR   = "plots"

RF_MODEL_PATH  = f"{MODEL_DIR}/rf_model.joblib"
MLP_MODEL_PATH = f"{MODEL_DIR}/mlp_model.joblib"   # StandardScaler 포함 Pipeline
META_PATH      = f"{MODEL_DIR}/meta.joblib"        # 클래스/피처 순서 등

RANDOM_SEED = 42
