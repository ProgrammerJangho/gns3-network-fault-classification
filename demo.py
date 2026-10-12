# -*- coding: utf-8 -*-
"""
demo.py
-------------------------------------------------------------------
저장된 모델로 실시간 장애 분류 데모.

두 가지 모드:
  1) GNS3 모드: 코어링크에 필터 1개 주입 -> VPCS ping 측정 -> 예측 -> 정상 복구
       python demo.py --class congestion
       python demo.py --class congestion --model mlp
  2) 합성 모드(GNS3 불필요, 발표 백업용): 해당 클래스 1샘플을 시뮬레이션 후 예측
       python demo.py --class high_delay --synth

출력: 예측 클래스(한글) + 클래스별 확률.
"""

import argparse
import time

import joblib
import numpy as np

import config
from features import compute_features, parse_vpcs_ping, to_vector


def load_model(which):
    path = config.RF_MODEL_PATH if which == "rf" else config.MLP_MODEL_PATH
    model = joblib.load(path)
    meta = joblib.load(config.META_PATH)
    return model, meta


def predict_and_report(model, meta, feat: dict, which: str):
    order = meta["features"]
    x = np.array([to_vector(feat, order)], dtype=float)
    pred = model.predict(x)[0]

    print("\n--- 측정 피처 ---")
    for k in order:
        print(f"  {k:10s} {feat[k]:.2f}")

    print(f"\n>>> 예측 [{which.upper()}]: {pred}  ({config.KOR_NAMES.get(pred, pred)})")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)
        print("    클래스별 확률:")
        for cls in config.CLASSES:
            if cls in classes:
                p = proba[classes.index(cls)]
                bar = "█" * int(round(p * 20))
                print(f"      {config.KOR_NAMES[cls]:<6} {p*100:5.1f}% {bar}")
    return pred


def run_synth(cls, which):
    """GNS3 없이 합성 1샘플 -> 예측 (발표 백업)."""
    from synth_data import simulate_episode
    rng = np.random.default_rng()
    rtts, sent = simulate_episode(config.SYNTH_PROFILES[cls], rng)
    feat = compute_features(rtts, sent)
    model, meta = load_model(which)
    print(f"[demo/synth] 주입 클래스: {cls} ({config.KOR_NAMES[cls]})")
    pred = predict_and_report(model, meta, feat, which)
    ok = "✅ 일치" if pred == cls else "❌ 불일치"
    print(f"\n[demo/synth] 정답={cls}, 예측={pred} -> {ok}")


def run_gns3(cls, which):
    """코어링크에 cls 필터 주입 -> 실측 -> 예측 -> 정상 복구."""
    from collect import GNS3Client, VPCSConsole

    gns = GNS3Client()
    pid = gns.resolve_project_id()
    link_id = gns.find_core_link_id(pid)
    chost, cport = gns.find_vpcs_console(pid, config.SRC_VPCS_NAME)

    from collect import warmup_bridge
    warmup = lambda: warmup_bridge(chost, cport, config.DST_VPCS_IP)

    print(f"[demo/gns3] 주입 클래스: {cls} ({config.KOR_NAMES[cls]}), "
          f"필터={config.FILTERS[cls]}")
    print("[demo/gns3] uBridge 워밍업 중...")
    warmup()
    try:
        gns.apply_filter(pid, link_id, config.FILTERS[cls], warmup=warmup)
        time.sleep(config.FILTER_SETTLE_S)
        with VPCSConsole(chost, cport) as con:
            out = con.ping(config.DST_VPCS_IP, config.PING_COUNT,
                           config.CONSOLE_READ_MAX_WAIT, config.CONSOLE_IDLE_BREAK_S)
        feat = compute_features(parse_vpcs_ping(out), config.PING_COUNT)
    finally:
        gns.apply_filter(pid, link_id, config.FILTERS["normal"], warmup=warmup)
        print("[demo/gns3] 코어링크 정상 복구 완료")

    model, meta = load_model(which)
    pred = predict_and_report(model, meta, feat, which)
    ok = "✅ 일치" if pred == cls else "⚠️ 불일치(실측 노이즈/타이밍 확인)"
    print(f"\n[demo/gns3] 주입={cls}, 예측={pred} -> {ok}")


def main():
    ap = argparse.ArgumentParser(description="실시간 장애 분류 데모")
    ap.add_argument("--class", dest="cls", required=True, choices=config.CLASSES,
                    help="주입할 장애 클래스")
    ap.add_argument("--model", choices=["rf", "mlp"], default="rf")
    ap.add_argument("--synth", action="store_true", help="GNS3 없이 합성 1샘플로 데모")
    args = ap.parse_args()

    if args.synth:
        run_synth(args.cls, args.model)
    else:
        run_gns3(args.cls, args.model)


if __name__ == "__main__":
    main()
