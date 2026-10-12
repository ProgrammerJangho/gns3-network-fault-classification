# -*- coding: utf-8 -*-
"""
features.py
-------------------------------------------------------------------
ping 텔레메트리 -> 피처 벡터 변환 로직 한 곳에 모음.
collect.py(실측), synth_data.py(합성), demo.py(추론) 가 전부 이 함수를 씀.
-> "수집/합성/데모에서 피처가 다르게 계산되는" 버그를 원천 차단.
"""

import re
import numpy as np

import config

# VPCS 정상 응답 줄 예: "84 bytes from 10.0.2.2 icmp_seq=1 ttl=60 time=120.4 ms"
# 실패(파싱 제외) 줄 예:
#   "10.0.2.2 icmp_seq=1 timeout"                              <- time= 없음
#   "*10.0.1.1 icmp_seq=2 ttl=255 time=15.3 ms (ICMP type:3, code:1, ...unreachable)"
#     ^ 이 줄에도 time= 가 있어서 정상으로 오인하면 안 됨!
_RTT_RE = re.compile(r"time=([0-9]+\.?[0-9]*)\s*ms")
_ERR_MARKERS = ("unreachable", "icmp type", "(icmp")


def parse_vpcs_ping(output: str):
    """VPCS ping 출력에서 '진짜 echo reply' 의 RTT(ms) 리스트만 추출.

    - 정상 응답 줄은 'bytes from' 을 포함함 -> 이것만 카운트.
    - 'Destination host unreachable' 같은 ICMP 에러 줄에도 time= 가 있지만,
      'bytes from' 이 없고 ICMP 에러 표식이 있으므로 제외(=손실로 처리).
    - timeout 줄은 time= 자체가 없어 자연히 제외.
    """
    rtts = []
    for line in output.splitlines():
        low = line.lower()
        if "time=" not in low:
            continue
        if "bytes from" not in low:          # 정상 reply 가 아님(에러/기타)
            continue
        if any(m in low for m in _ERR_MARKERS):
            continue
        m = _RTT_RE.search(line)
        if m:
            rtts.append(float(m.group(1)))
    return rtts


def compute_features(rtts, sent, timeout_ceiling=config.TIMEOUT_CEILING_MS):
    """RTT 리스트 -> 7개 피처 dict.

    rtts : 수신된 패킷들의 RTT(ms) 리스트 (손실 패킷은 미포함)
    sent : 보낸 패킷 수 (보통 PING_COUNT)

    전 패킷 손실(수신 0개)이면 RTT 정보가 없으므로 timeout 천장값으로 채움
    -> 링크다운의 명확한 시그니처(loss 100% + RTT 천장).
    """
    n_recv = len(rtts)
    loss_pct = 100.0 * (sent - n_recv) / sent if sent > 0 else 100.0

    if n_recv == 0:
        return {
            "loss_pct":  loss_pct,
            "rtt_mean":  timeout_ceiling,
            "rtt_std":   0.0,
            "rtt_min":   timeout_ceiling,
            "rtt_max":   timeout_ceiling,
            "rtt_range": 0.0,
            "jitter":    0.0,
        }

    arr = np.asarray(rtts, dtype=float)
    if n_recv >= 2:
        rtt_std = float(np.std(arr))                       # 모표준편차(ddof=0)
        jitter = float(np.mean(np.abs(np.diff(arr))))      # 연속 RTT 차의 평균 절대값
    else:
        rtt_std = 0.0
        jitter = 0.0

    return {
        "loss_pct":  loss_pct,
        "rtt_mean":  float(arr.mean()),
        "rtt_std":   rtt_std,
        "rtt_min":   float(arr.min()),
        "rtt_max":   float(arr.max()),
        "rtt_range": float(arr.max() - arr.min()),
        "jitter":    jitter,
    }


def to_vector(feat: dict, order=None):
    """피처 dict -> 모델 입력용 리스트(고정 순서)."""
    if order is None:
        order = config.FEATURES
    return [feat[k] for k in order]


def feature_row(rtts, sent, label, timeout_ceiling=config.TIMEOUT_CEILING_MS):
    """CSV 한 줄용 dict: 피처 + label.  컬럼 순서는 config.FEATURES 를 따름."""
    feats = compute_features(rtts, sent, timeout_ceiling)
    row = {f: feats[f] for f in config.FEATURES}
    row["label"] = label
    return row
