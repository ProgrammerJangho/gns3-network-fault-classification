# -*- coding: utf-8 -*-
"""
collect.py
-------------------------------------------------------------------
GNS3 실측 수집기.

흐름:
  1) GNS3 REST API 로 프로젝트/코어링크(P1-P2)/VPCS 콘솔포트 자동 탐색
  2) 클래스별로 코어링크에 패킷 필터 주입 (PUT .../links/{id}, body={"filters":...})
     - 샘플마다 수치를 살짝 흔들어 다양화
  3) VPCS1 콘솔(telnet)에서 VPCS2 로 ping -c N -> 패킷별 RTT 파싱
  4) features.compute_features 로 벡터화 -> dataset_raw.csv
  5) 끝나면 코어링크를 정상({}) 으로 복구

v2.2 와 v3(JWT 인증) 모두 지원. config.GNS3_API_VERSION 으로 전환.

사용:
  python collect.py --inspect          # 노드/링크/콘솔포트만 출력(설정 채우기용)
  python collect.py                     # 수집 실행
  python collect.py --samples 40        # 클래스당 개수 변경
  python collect.py --restore           # 코어링크 정상 복구만
"""

import argparse
import copy
import socket
import time

import pandas as pd
import requests

import config
from features import feature_row


# =====================================================================
# GNS3 REST 클라이언트 (v2 / v3)
# =====================================================================
class GNS3Client:
    def __init__(self):
        self.base = f"{config.GNS3_SCHEME}://{config.GNS3_HOST}:{config.GNS3_PORT}/{config.GNS3_API_VERSION}"
        self.sess = requests.Session()
        if config.GNS3_API_VERSION == "v3":
            self._login_v3()
        elif config.GNS3_API_VERSION == "v2" and getattr(config, "GNS3_AUTH", False):
            # GNS3 2.2 "Protect server with password" 사용 시 HTTP Basic Auth
            self.sess.auth = (config.GNS3_USERNAME, config.GNS3_PASSWORD)

    def _login_v3(self):
        # POST /v3/users/login  (form-data: username, password) -> access_token
        url = f"{config.GNS3_SCHEME}://{config.GNS3_HOST}:{config.GNS3_PORT}/v3/users/login"
        r = self.sess.post(
            url,
            data={"username": config.GNS3_USERNAME, "password": config.GNS3_PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        self.sess.headers.update({"Authorization": f"Bearer {token}"})

    def _get(self, path):
        r = self.sess.get(self.base + path, timeout=10)
        r.raise_for_status()
        return r.json()

    def _put(self, path, body):
        r = self.sess.put(self.base + path, json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    # --- 리소스 탐색 ---
    def resolve_project_id(self):
        if config.GNS3_PROJECT_ID:
            return config.GNS3_PROJECT_ID
        for p in self._get("/projects"):
            if p.get("name") == config.GNS3_PROJECT_NAME:
                return p["project_id"]
        raise RuntimeError(f"프로젝트 '{config.GNS3_PROJECT_NAME}' 를 찾을 수 없음")

    def get_nodes(self, pid):
        return self._get(f"/projects/{pid}/nodes")

    def get_links(self, pid):
        return self._get(f"/projects/{pid}/links")

    def find_core_link_id(self, pid):
        """CORE_LINK_ENDPOINTS(예: P1,P2) 를 잇는 링크의 link_id."""
        id2name = {n["node_id"]: n["name"] for n in self.get_nodes(pid)}
        want = set(config.CORE_LINK_ENDPOINTS)
        for link in self.get_links(pid):
            names = {id2name.get(ep["node_id"]) for ep in link.get("nodes", [])}
            if want.issubset(names):
                return link["link_id"]
        raise RuntimeError(f"{config.CORE_LINK_ENDPOINTS} 를 잇는 링크를 못 찾음")

    def find_vpcs_console(self, pid, name):
        """VPCS 노드 이름 -> (console_host, console_port)."""
        for n in self.get_nodes(pid):
            if n["name"] == name:
                host = config.CONSOLE_HOST or n.get("console_host") or config.GNS3_HOST
                return host, n["console"]
        raise RuntimeError(f"VPCS 노드 '{name}' 를 못 찾음")

    def apply_filter(self, pid, link_id, filters, retries=5, warmup=None):
        """코어링크에 필터 주입 (정상은 {} = 필터 해제).

        dynamips 직결 링크는 트래픽이 없으면 uBridge 가 안 떠서 409
        ('bridge ... doesn't exist') 가 난다. 그 경우 warmup() 으로 ping 을
        한 번 흘려 uBridge 를 깨운 뒤 재시도한다.
        """
        path = f"/projects/{pid}/links/{link_id}"
        last = None
        for _ in range(retries):
            r = self.sess.put(self.base + path, json={"filters": filters}, timeout=15)
            if r.status_code in (200, 201):
                return r.json()
            last = r
            if r.status_code == 409:
                if warmup is not None:
                    try:
                        warmup()
                    except Exception:
                        pass
                time.sleep(2.0)
                continue
            r.raise_for_status()
        raise RuntimeError(
            f"필터 적용 실패(409 지속): {filters}\n"
            f"-> 모든 노드가 start 상태이고 PC1->PC2 ping 이 되는지 확인하세요.\n"
            f"   응답: {last.text[:200] if last is not None else ''}")


# =====================================================================
# VPCS 콘솔 (telnet) — socket 기반, IAC 협상 바이트 제거
# =====================================================================
_IAC = 0xFF


def _strip_telnet(data: bytes) -> bytes:
    """telnet IAC 명령 시퀀스 제거 (순수 텍스트만 남김)."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        if b == _IAC:
            if i + 1 >= n:
                break
            cmd = data[i + 1]
            if cmd == 0xFA:                      # SB ... IAC SE
                j = i + 2
                while j + 1 < n and not (data[j] == _IAC and data[j + 1] == 0xF0):
                    j += 1
                i = j + 2
            elif cmd in (0xFB, 0xFC, 0xFD, 0xFE):  # WILL/WONT/DO/DONT + option
                i += 3
            else:                                # 기타 2바이트 명령
                i += 2
        else:
            out.append(b)
            i += 1
    return bytes(out)


class VPCSConsole:
    def __init__(self, host, port, prompt=config.VPCS_PROMPT, connect_timeout=5.0):
        self.host, self.port = host, port
        self.prompt = prompt
        self.connect_timeout = connect_timeout
        self.sock = None

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), self.connect_timeout)
        self.sock.settimeout(0.5)
        self._drain(idle_s=1.0, max_s=3.0)   # 접속 배너 비우기
        return self

    def __exit__(self, *a):
        try:
            self.sock.close()
        except Exception:
            pass

    def _recv(self):
        try:
            return self.sock.recv(4096)
        except socket.timeout:
            return b""

    def _drain(self, idle_s, max_s):
        buf = bytearray()
        last = start = time.time()
        while True:
            chunk = self._recv()
            now = time.time()
            if chunk:
                buf += chunk
                last = now
            elif now - last >= idle_s:
                break
            if now - start >= max_s:
                break
        return _strip_telnet(bytes(buf)).decode("ascii", "ignore")

    def ping(self, ip, count, max_wait, idle_break):
        # 프롬프트가 보이도록 빈 줄 한 번
        self.sock.sendall(b"\r\n")
        self._drain(idle_s=0.4, max_s=1.5)

        self.sock.sendall(f"ping {ip} -c {count}\r\n".encode())

        buf = bytearray()
        start = last = time.time()
        while True:
            chunk = self._recv()
            now = time.time()
            if chunk:
                buf += chunk
                last = now
            text = _strip_telnet(bytes(buf)).decode("ascii", "ignore")
            # 모든 패킷의 응답 줄(icmp_seq=) 이 다 들어오면 종료
            if text.count("icmp_seq=") >= count:
                break
            # 출력이 멈췄고 프롬프트가 다시 떴으면 종료(타임아웃 줄을 안 찍는 VPCS 대비)
            if not chunk and (now - last) >= idle_break and self.prompt in text and len(buf) > 0:
                break
            if now - start >= max_wait:
                break
        return _strip_telnet(bytes(buf)).decode("ascii", "ignore")


# =====================================================================
# 필터 다양화
# =====================================================================
def warmup_bridge(chost, cport, ip, count=3):
    """짧은 ping 으로 코어 링크 uBridge 를 깨운다(409 대비)."""
    try:
        with VPCSConsole(chost, cport) as con:
            con.ping(ip, count, max_wait=20, idle_break=1.0)
    except Exception:
        pass


def jitter_filter(base_filter, frac, rng):
    """필터 수치를 ±frac 으로 흔듦. frequency_drop 은 그대로(=링크다운 시그니처 유지)."""
    f = copy.deepcopy(base_filter)
    if "delay" in f:
        lat, jit = f["delay"][0], f["delay"][1] if len(f["delay"]) > 1 else 0
        lat = max(0, int(round(lat * (1 + rng.uniform(-frac, frac)))))
        jit = max(0, int(round(jit * (1 + rng.uniform(-frac, frac)))))
        f["delay"] = [lat, jit]
    if "packet_loss" in f:
        pl = f["packet_loss"][0]
        if pl >= 100:
            f["packet_loss"] = [100]   # 링크다운(100% 손실)은 흔들지 않고 고정
        else:
            pl = min(100, max(0, int(round(pl * (1 + rng.uniform(-frac, frac))))))
            f["packet_loss"] = [pl]
    return f


# =====================================================================
# 수집
# =====================================================================
def collect(samples_per_class, out_path, only=None):
    import numpy as np
    rng = np.random.default_rng(config.RANDOM_SEED)

    classes = only if only else config.CLASSES

    gns = GNS3Client()
    pid = gns.resolve_project_id()
    link_id = gns.find_core_link_id(pid)
    chost, cport = gns.find_vpcs_console(pid, config.SRC_VPCS_NAME)

    print(f"[collect] project={pid}")
    print(f"[collect] core link({'-'.join(config.CORE_LINK_ENDPOINTS)})={link_id}")
    print(f"[collect] {config.SRC_VPCS_NAME} console={chost}:{cport} -> ping {config.DST_VPCS_IP}")
    print(f"[collect] 수집 클래스: {classes}")

    # uBridge 워밍업 콜백 (409 시 ping 흘려 깨움)
    warmup = lambda: warmup_bridge(chost, cport, config.DST_VPCS_IP)
    print("[collect] uBridge 워밍업 중...")
    warmup()

    rows = []
    try:
        for cls in classes:
            base = config.FILTERS[cls]
            print(f"\n[collect] === {cls} ({config.KOR_NAMES[cls]}) x{samples_per_class} ===")
            for i in range(samples_per_class):
                filt = jitter_filter(base, config.FILTER_JITTER_FRAC, rng)
                gns.apply_filter(pid, link_id, filt, warmup=warmup)
                time.sleep(config.FILTER_SETTLE_S)

                with VPCSConsole(chost, cport) as con:
                    out = con.ping(config.DST_VPCS_IP, config.PING_COUNT,
                                   config.CONSOLE_READ_MAX_WAIT,
                                   config.CONSOLE_IDLE_BREAK_S)

                from features import parse_vpcs_ping
                rtts = parse_vpcs_ping(out)
                row = feature_row(rtts, config.PING_COUNT, cls)
                rows.append(row)
                print(f"  [{i+1:>2}/{samples_per_class}] filt={filt} "
                      f"loss={row['loss_pct']:.0f}% rtt_mean={row['rtt_mean']:.1f} "
                      f"jitter={row['jitter']:.1f}")
    finally:
        # 무슨 일이 있어도 링크는 정상 복구
        gns.apply_filter(pid, link_id, config.FILTERS["normal"], warmup=warmup)
        print("\n[collect] 코어링크 정상({}) 복구 완료")

    df = pd.DataFrame(rows, columns=config.FEATURES + ["label"])
    df.to_csv(out_path, index=False)
    print(f"[collect] 저장: {out_path} (총 {len(df)}행)")
    print(df.groupby("label")[config.FEATURES].mean().reindex(classes).round(2).to_string())


def inspect():
    gns = GNS3Client()
    pid = gns.resolve_project_id()
    print(f"project_id = {pid}")
    print("\n[nodes]")
    for n in gns.get_nodes(pid):
        host = n.get("console_host", "-")
        print(f"  {n['name']:<8} type={n.get('node_type'):<12} console={host}:{n.get('console')}")
    print("\n[links]")
    id2name = {n["node_id"]: n["name"] for n in gns.get_nodes(pid)}
    for link in gns.get_links(pid):
        eps = " - ".join(id2name.get(ep["node_id"], "?") for ep in link.get("nodes", []))
        print(f"  {link['link_id']}  ({eps})")


def restore():
    gns = GNS3Client()
    pid = gns.resolve_project_id()
    link_id = gns.find_core_link_id(pid)
    chost, cport = gns.find_vpcs_console(pid, config.SRC_VPCS_NAME)
    warmup = lambda: warmup_bridge(chost, cport, config.DST_VPCS_IP)
    gns.apply_filter(pid, link_id, config.FILTERS["normal"], warmup=warmup)
    print("[collect] 코어링크 정상 복구 완료")


def main():
    ap = argparse.ArgumentParser(description="GNS3 실측 수집")
    ap.add_argument("--inspect", action="store_true", help="노드/링크/콘솔포트 출력")
    ap.add_argument("--restore", action="store_true", help="코어링크 정상 복구만")
    ap.add_argument("--samples", type=int, default=config.SAMPLES_PER_CLASS)
    ap.add_argument("--only", nargs="+", choices=config.CLASSES,
                    help="특정 클래스만 수집 (예: --only congestion)")
    ap.add_argument("--out", default=None, help="출력 CSV (기본 dataset_raw.csv)")
    args = ap.parse_args()

    if args.inspect:
        inspect()
    elif args.restore:
        restore()
    else:
        # --only 사용 시 기본 출력을 별도 파일로(전체 dataset_raw.csv 덮어쓰기 방지)
        if args.out:
            out = args.out
        elif args.only:
            out = f"dataset_only_{'_'.join(args.only)}.csv"
        else:
            out = config.RAW_CSV
        collect(args.samples, out, only=args.only)


if __name__ == "__main__":
    main()
