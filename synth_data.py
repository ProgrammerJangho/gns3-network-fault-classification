# -*- coding: utf-8 -*-
"""
synth_data.py
-------------------------------------------------------------------
GNS3 없이 train 파이프라인을 먼저 검증하기 위한 합성 데이터 생성기.

핵심: 클래스별 프로파일(config.SYNTH_PROFILES)대로 '패킷별 RTT'를 직접
시뮬레이션하고, features.compute_features 로 벡터화한다.
-> 실측 collect.py 와 완전히 동일한 피처 경로를 타므로,
   여기서 train 이 잘 돌면 GNS3 데이터로 바꿔도 그대로 동작.

출력: config.RAW_CSV (collect.py 가 만드는 것과 같은 포맷)
사용: python synth_data.py [--out dataset_raw.csv] [--per-class 30] [--seed 42]
"""

import argparse
import numpy as np
import pandas as pd

import config
from features import feature_row


def _jitter(val, frac, rng):
    """val 을 ±frac 비율로 흔든 값."""
    return val * (1.0 + rng.uniform(-frac, frac))


def simulate_episode(profile, rng):
    """프로파일 1개 -> ping PING_COUNT 회의 패킷별 RTT 리스트 + 보낸 수.

    - 샘플마다 extra_delay / jitter_std / loss 를 살짝 흔들어 다양화
    - loss=0 또는 1(정상/링크다운)의 극단값은 흔들지 않아 시그니처 유지
    """
    extra = profile["extra_delay"]
    jit_std = profile["jitter_std"]
    loss = profile["loss"]

    if extra > 0:
        extra = _jitter(extra, config.SYNTH_JITTER_FRAC, rng)
    if jit_std > 0:
        jit_std = _jitter(jit_std, config.SYNTH_JITTER_FRAC, rng)
    if 0.0 < loss < 1.0:
        loss = min(1.0, max(0.0, _jitter(loss, config.SYNTH_JITTER_FRAC, rng)))

    base = _jitter(config.BASE_RTT_MS, 0.1, rng)
    sent = config.PING_COUNT

    rtts = []
    for _ in range(sent):
        if rng.random() < loss:
            continue  # 패킷 손실 -> 수신 안 됨
        val = base + extra + rng.normal(0.0, jit_std)
        rtts.append(max(0.1, float(val)))   # RTT 는 양수
    return rtts, sent


def generate(per_class, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for cls in config.CLASSES:
        profile = config.SYNTH_PROFILES[cls]
        for _ in range(per_class):
            rtts, sent = simulate_episode(profile, rng)
            rows.append(feature_row(rtts, sent, cls))
    df = pd.DataFrame(rows, columns=config.FEATURES + ["label"])
    return df


def main():
    ap = argparse.ArgumentParser(description="합성 장애 데이터 생성")
    ap.add_argument("--out", default=config.RAW_CSV)
    ap.add_argument("--per-class", type=int, default=config.SAMPLES_PER_CLASS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    df = generate(args.per_class, args.seed)
    df.to_csv(args.out, index=False)

    print(f"[synth] 저장: {args.out}  (총 {len(df)}행, 클래스당 {args.per_class}개)")
    print("[synth] 클래스별 평균 시그니처:")
    summary = df.groupby("label")[config.FEATURES].mean().round(2)
    # 클래스 순서 고정 출력
    summary = summary.reindex(config.CLASSES)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(summary.to_string())


if __name__ == "__main__":
    main()
