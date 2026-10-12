# -*- coding: utf-8 -*-
"""
augment.py
-------------------------------------------------------------------
실측(또는 합성) 원본을 클래스별로 증강.

방법(스펙대로):
  클래스별로 각 피처의 (평균, 표준편차)를 관측값에서 추정 ->
  그 분산에 맞춘 가우시안 노이즈로 N개까지 증강 -> 유효 범위로 clip.

추가 보정(데이터 일관성):
  - rtt_min <= rtt_max 가 깨지면 swap
  - rtt_range = rtt_max - rtt_min 으로 재계산
  -> 피처 간 모순(예: min>max) 없는 깨끗한 증강 샘플.

입력 : config.RAW_CSV
출력 : config.DATASET_CSV  (원본 + 증강, train/demo 가 사용)
사용 : python augment.py [--in dataset_raw.csv] [--out dataset.csv]
                         [--per-class 200] [--scale 1.0] [--seed 42]
"""

import argparse
import numpy as np
import pandas as pd

import config


def _clip(value, bounds):
    lo, hi = bounds
    return float(min(hi, max(lo, value)))


def _fix_consistency(row: dict) -> dict:
    """min/max/range 간 모순 제거."""
    lo, hi = sorted([row["rtt_min"], row["rtt_max"]])
    row["rtt_min"], row["rtt_max"] = lo, hi
    row["rtt_range"] = hi - lo
    return row


def augment_class(sub: pd.DataFrame, cls: str, target_n: int,
                  scale: float, rng) -> pd.DataFrame:
    """한 클래스의 원본 sub -> 총 target_n 개 (원본 유지 + 가우시안 증강)."""
    mu = sub[config.FEATURES].mean()
    sd = sub[config.FEATURES].std(ddof=0).fillna(0.0)   # 표본 1개 등으로 NaN 나면 0

    n_keep = len(sub)
    n_new = max(0, target_n - n_keep)

    new_rows = []
    for _ in range(n_new):
        row = {}
        for f in config.FEATURES:
            noise = rng.normal(0.0, sd[f] * scale)
            row[f] = _clip(mu[f] + noise, config.FEATURE_BOUNDS[f])
        row = _fix_consistency(row)
        row["label"] = cls
        new_rows.append(row)

    parts = [sub.copy()]
    if new_rows:
        parts.append(pd.DataFrame(new_rows, columns=config.FEATURES + ["label"]))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="클래스별 가우시안 증강")
    ap.add_argument("--in", dest="inp", default=config.RAW_CSV)
    ap.add_argument("--out", default=config.DATASET_CSV)
    ap.add_argument("--per-class", type=int, default=config.N_PER_CLASS)
    ap.add_argument("--scale", type=float, default=config.AUG_NOISE_SCALE)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = pd.read_csv(args.inp)

    out_parts = []
    for cls in config.CLASSES:
        sub = raw[raw["label"] == cls].reset_index(drop=True)
        if len(sub) == 0:
            print(f"[augment] 경고: '{cls}' 원본 0개 -> 건너뜀")
            continue
        out_parts.append(augment_class(sub, cls, args.per_class, args.scale, rng))

    out = pd.concat(out_parts, ignore_index=True)
    # 셔플(학습 시 클래스가 뭉쳐있지 않도록)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print(f"[augment] 저장: {args.out}  (총 {len(out)}행)")
    print("[augment] 클래스별 개수:")
    print(out["label"].value_counts().reindex(config.CLASSES).to_string())
    print("\n[augment] 증강 후 클래스별 평균 시그니처:")
    summary = out.groupby("label")[config.FEATURES].mean().reindex(config.CLASSES).round(2)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(summary.to_string())


if __name__ == "__main__":
    main()
