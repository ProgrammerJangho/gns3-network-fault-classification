# -*- coding: utf-8 -*-
"""
train.py
-------------------------------------------------------------------
RF(메인) + MLP(비교) 학습 · 평가 · 그래프 · 모델 저장.

- 5-fold Stratified CV 로 accuracy / macro-F1 (mean±std)
- cross_val_predict 로 confusion matrix (각 모델)
- RF feature importance
- 모델 비교 막대그래프
- 전체 데이터로 재학습 후 joblib 저장 (demo.py 가 로드)

그래프 라벨은 ASCII 클래스 키 사용(한글 폰트 깨짐 방지).
입력 : config.DATASET_CSV
사용 : python train.py [--data dataset.csv]
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # 헤드리스 환경에서 PNG 저장
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict)
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score)

import config


def build_models():
    """RF, MLP(스케일러 포함 파이프라인) 반환."""
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=2000,
            random_state=config.RANDOM_SEED,
        )),
    ])
    return {"RandomForest": rf, "MLP": mlp}


def evaluate(name, model, X, y, cv):
    """CV 점수 + cross_val_predict 예측 반환."""
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    acc = scores["test_accuracy"]
    f1 = scores["test_f1_macro"]
    print(f"\n[{name}] 5-fold CV")
    print(f"  accuracy : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  macro-F1 : {f1.mean():.4f} ± {f1.std():.4f}")
    return {
        "acc_mean": acc.mean(), "acc_std": acc.std(),
        "f1_mean": f1.mean(),  "f1_std": f1.std(),
        "y_pred": y_pred,
    }


def plot_confusion(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred, labels=config.CLASSES)
    plt.figure(figsize=(6.2, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES,
                cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_feature_importance(rf, path):
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    feats = [config.FEATURES[i] for i in order]
    vals = imp[order]
    plt.figure(figsize=(7, 4.2))
    sns.barplot(x=vals, y=feats, color="#4c72b0")
    plt.title("Feature Importance (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_model_comparison(results, path):
    names = list(results.keys())
    acc = [results[n]["acc_mean"] for n in names]
    f1 = [results[n]["f1_mean"] for n in names]
    acc_err = [results[n]["acc_std"] for n in names]
    f1_err = [results[n]["f1_std"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(6.2, 4.4))
    plt.bar(x - w / 2, acc, w, yerr=acc_err, capsize=4, label="Accuracy", color="#4c72b0")
    plt.bar(x + w / 2, f1, w, yerr=f1_err, capsize=4, label="macro-F1", color="#dd8452")
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    for xi, a, f in zip(x, acc, f1):
        plt.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        plt.text(xi + w / 2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Score")
    plt.title("Model Comparison (5-fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="RF/MLP 학습·평가")
    ap.add_argument("--data", default=config.DATASET_CSV)
    args = ap.parse_args()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df[config.FEATURES].values
    y = df["label"].values
    print(f"[train] 데이터: {args.data}  ({len(df)}행, {len(config.FEATURES)}피처, "
          f"{len(set(y))}클래스)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    models = build_models()

    # MLP 수렴 경고는 결과에 영향 없으면 조용히
    warnings.filterwarnings("ignore", category=UserWarning)

    results = {}
    for name, model in models.items():
        results[name] = evaluate(name, model, X, y, cv)

    # confusion matrix (각 모델, CV 예측 기준)
    for name in models:
        path = f"{config.PLOTS_DIR}/cm_{name.lower()}.png"
        plot_confusion(y, results[name]["y_pred"], name, path)
        print(f"[train] 저장: {path}")

    # 모델 비교 그래프
    cmp_path = f"{config.PLOTS_DIR}/model_comparison.png"
    plot_model_comparison(results, cmp_path)
    print(f"[train] 저장: {cmp_path}")

    # 전체 데이터로 재학습 -> 저장 + RF 중요도
    rf_full = build_models()["RandomForest"].fit(X, y)
    mlp_full = build_models()["MLP"].fit(X, y)

    fi_path = f"{config.PLOTS_DIR}/feature_importance.png"
    plot_feature_importance(rf_full, fi_path)
    print(f"[train] 저장: {fi_path}")

    joblib.dump(rf_full, config.RF_MODEL_PATH)
    joblib.dump(mlp_full, config.MLP_MODEL_PATH)
    joblib.dump({"classes": config.CLASSES, "features": config.FEATURES},
                config.META_PATH)
    print(f"[train] 모델 저장: {config.RF_MODEL_PATH}, {config.MLP_MODEL_PATH}")

    # RF 상세 리포트(메인 모델)
    print("\n[RandomForest] classification report (CV 예측 기준)")
    print(classification_report(y, results["RandomForest"]["y_pred"],
                                labels=config.CLASSES, digits=3, zero_division=0))

    # RF feature importance 텍스트
    imp = rf_full.feature_importances_
    print("[RandomForest] feature importance")
    for f, v in sorted(zip(config.FEATURES, imp), key=lambda t: -t[1]):
        print(f"  {f:10s} {v:.4f}")


if __name__ == "__main__":
    main()
