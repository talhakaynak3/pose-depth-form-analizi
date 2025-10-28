import os, json, math, argparse, statistics
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image, ImageDraw, ImageFont



# ================= CLI =================
def parse_args():
    p = argparse.ArgumentParser("Pose + Depth pipeline (ref/eval)")
    p.add_argument("--mode", choices=["ref", "eval"], required=True,
                   help="ref: referans üret | eval: videoyu referansa göre değerlendir")
    p.add_argument("--src", required=True, help="Giriş videosu (ref veya test video)")
    p.add_argument("--refjson", default="reference_correct.json",
                   help="Referans JSON yolu (ref çıktısı / eval girdisi)")
    p.add_argument("--out", default="out_overlay.mp4", help="Overlay’li çıktı videosu")
    p.add_argument("--calib", default="", help="(opsiyonel) omuz/dirsek ofsetleri JSON")
    p.add_argument("--save_metrics", default="", help="(opsiyonel) ham metrikleri JSON’a kaydet")
    p.add_argument("--show", action="store_true", help="Pencere göster")
    return p.parse_args()

ARGS = parse_args()

# ============ Kalibrasyon ofsetleri (opsiyonel) ============
CALIB = {"shoulder_offset_xy": [0, 0], "elbow_offset_xy": [0, 0]}
if ARGS.calib and Path(ARGS.calib).exists():
    with open(ARGS.calib, "r", encoding="utf-8") as f:
        CALIB.update(json.load(f))
    print("[CALIB] yüklendi:", CALIB)
else:
    print("[CALIB] kullanılmıyor (0,0).")

def apply_offset(idx, u, v):
    # MediaPipe indexleri:
    # 11: L-shoulder, 12: R-shoulder, 13: L-elbow, 14: R-elbow
    if idx in (11, 12):
        dx, dy = CALIB["shoulder_offset_xy"]; return u+dx, v+dy
    if idx in (13, 14):
        dx, dy = CALIB["elbow_offset_xy"];    return u+dx, v+dy
    return u, v

# ============ MiDaS (depth) ============
print("[MiDaS] yükleniyor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
print("[MiDaS] hazır:", device)

# ============ MediaPipe Pose ============
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawer = mp.solutions.drawing_utils
style  = mp.solutions.drawing_styles

# ============ Yardımcılar ============
def angle_3pts(a, b, c):
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    dot = bax*bcx + bay*bcy
    den = (math.hypot(bax, bay) * math.hypot(bcx, bcy)) + 1e-6
    v = max(-1.0, min(1.0, dot/den))
    return math.degrees(math.acos(v))

def to_px(lm, w, h): return int(lm.x*w), int(lm.y*h)

def open_video(path):
    if not Path(path).exists(): raise SystemExit(f"[HATA] video yok: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():      raise SystemExit(f"[HATA] video açılamadı: {path}")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps= cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(ARGS.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    return cap, writer, w, h, fps

def percentile_range(vals, lo=10, hi=90):
    if not vals: return [None, None]
    arr = np.array(vals, dtype=float)
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

def within(val, lo, hi):
    return (lo is None or val >= lo) and (hi is None or val <= hi)

# === FEEDBACK: referans dışı durumları metne çevir ===
def feedback_text(metrics_item, ranges):
    """
    metrics_item: {"elbow_left":..,"elbow_right":..,"shoulder_left":..,"shoulder_right":..,"z":..}
    ranges: {"elbow_left":[lo,hi], ...}
    return: list[str] (kısa uyarılar)
    """
    msgs = []
    if not ranges:
        return msgs

    def check(val, lo, hi, nice):
        if val is None or lo is None or hi is None:
            return
        if val < lo:
            msgs.append(f"{nice} fazla kapalı ({val:.0f}° < {lo:.0f}°)")
        elif val > hi:
            msgs.append(f"{nice} fazla açık ({val:.0f}° > {hi:.0f}°)")

    # açı uyarıları
    check(metrics_item.get("elbow_left"),   *ranges.get("elbow_left",   [None, None]), "Sol dirsek")
    check(metrics_item.get("elbow_right"),  *ranges.get("elbow_right",  [None, None]), "Sağ dirsek")
    check(metrics_item.get("shoulder_left"),*ranges.get("shoulder_left",[None, None]), "Sol omuz")
    check(metrics_item.get("shoulder_right"),*ranges.get("shoulder_right",[None, None]),"Sağ omuz")

    # --- BAR ve SHOULDER PROTR geri bildirimleri (margin'li) ---
    MARGIN_BAR = 0.02
    MARGIN_PRO = 0.02

    # Bar aşağı/yukarı (margin'li)
    if "bar_drop" in metrics_item and "bar_drop" in ranges:
        val = metrics_item["bar_drop"]
        lo, hi = ranges["bar_drop"]
        if val is not None:
            if val > hi + MARGIN_BAR:
                msgs.append(f"Bar çok aşağı indi ({val:.3f} > {hi:.3f})")
            elif val < lo - MARGIN_BAR:
                msgs.append(f"Bar çok yukarıda ({val:.3f} < {lo:.3f})")

     #Omuz öne eğilme (margin'li)
    if "shoulder_protr" in metrics_item and "shoulder_protr" in ranges:
        val = metrics_item["shoulder_protr"]
        lo, hi = ranges["shoulder_protr"]
        if val is not None:
            if val > hi + MARGIN_PRO:
                msgs.append(f"Omuz fazla öne eğildi ({val:.3f} > {hi:.3f})")
            elif val < lo - MARGIN_PRO:
                msgs.append(f"Omuz çok geride ({val:.3f} < {lo:.3f})")

    # Bar çok aşağıda veya yukarıda mı?
    if "bar_drop" in metrics_item and "bar_drop" in ranges:
        lo, hi = ranges["bar_drop"]
        if metrics_item["bar_drop"] is not None:
            if metrics_item["bar_drop"] > hi:
                msgs.append(f"Bar çok aşağı indi ({metrics_item['bar_drop']:.3f} > {hi:.3f})")
            elif metrics_item["bar_drop"] < lo:
                msgs.append(f"Bar çok yukarıda ({metrics_item['bar_drop']:.3f} < {lo:.3f})")

    # Omuz fazla öne mi eğildi?
    if "shoulder_protr" in metrics_item and "shoulder_protr" in ranges:
        lo, hi = ranges["shoulder_protr"]
        if metrics_item["shoulder_protr"] is not None:
            if metrics_item["shoulder_protr"] > hi:
                msgs.append(f"Omuz fazla öne eğildi ({metrics_item['shoulder_protr']:.3f} > {hi:.3f})")
            elif metrics_item["shoulder_protr"] < lo:
                msgs.append(f"Omuz çok geride ({metrics_item['shoulder_protr']:.3f} < {lo:.3f})")

    # Z (relative depth) – yön modele göre değişebildiği için nötr mesaj
    z = metrics_item.get("z")
    z_lo, z_hi = ranges.get("z", [None, None])
    if z is not None and z_lo is not None and z_hi is not None:
        if z < z_lo or z > z_hi:
            msgs.append("Z (derinlik) referansın dışında (çok yakın/uzak)")


    return msgs

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"   # veya segoeui.ttf, DejaVuSans.ttf
FONT_INFO = ImageFont.truetype(FONT_PATH, 22)
FONT_WARN = ImageFont.truetype(FONT_PATH, 24)


# ============ Çekirdek işlev: bir videodan metrik çıkar ============
def extract_metrics(video_path, render_overlay=True, ranges=None):
    """
    ranges verilirse (eval modunda) dışına çıkanları işaretler.
    Döndürür: metrics(list of dict per frame), issues_counter(dict)
    """
    cap, writer, W, H, FPS = open_video(video_path)
    print(f"[OK] {video_path} | {W}x{H}@{FPS:.1f}")

    L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11,12,13,14,15,16,23,24
    metrics = []
    issues_counter = {"elbow_left": 0, "elbow_right": 0, "shoulder_left": 0, "shoulder_right": 0,
                      "z": 0, "bar_drop": 0, "shoulder_protr": 0}

    # --- SMOOTHING PARAMS ---
    ALPHA_BAR = 0.3  # 0.2–0.4 arası iyi
    ALPHA_PRO = 0.3
    ema_bar = None  # Exponential Moving Average buffer
    ema_pro = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]


        # depth
        rgb_m = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp   = transforms(rgb_m).to(device)
        with torch.no_grad():
            pred = midas(inp).squeeze().cpu().numpy()
        depth = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)

        # pose
        res = pose.process(rgb_m)
        item = {"elbow_left":None,"elbow_right":None,"shoulder_left":None,"shoulder_right":None,"z":None}


        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # piksel koordinatları
            def P(i):
                u, v = to_px(lm[i], w, h); u, v = apply_offset(i, u, v); return (u, v)

            l_sh, r_sh = P(L_SH), P(R_SH)
            l_el, r_el = P(L_EL), P(R_EL)
            l_wr, r_wr = P(L_WR), P(R_WR)
            l_hip,r_hip= P(L_HIP),P(R_HIP)

            # açı hesapları
            L_elb = angle_3pts(l_sh, l_el, l_wr)
            R_elb = angle_3pts(r_sh, r_el, r_wr)
            L_shd = angle_3pts(l_hip, l_sh, l_el)
            R_shd = angle_3pts(r_hip, r_sh, r_el)

            # Z metriği: iki omuz derinliğinin ortalaması (relative)
            z_vals = []
            for (u,v) in [l_sh, r_sh]:
                if 0<=u<w and 0<=v<h: z_vals.append(float(depth[v, u]))
            Zrel = float(np.mean(z_vals)) if z_vals else None
            # --- BAR DROP (bilek ortalaması vs omuz ortalaması) ---
            bar_y_norm = ((l_wr[1] + r_wr[1]) / 2) / h
            sh_y_norm = ((l_sh[1] + r_sh[1]) / 2) / h
            bar_drop = bar_y_norm - sh_y_norm # + ise bar omuz çizgisinin ALTINDA

            # --- DEBUG GÖRSELİ: omuz ve bar çizgileri + sayı ---
            #mid_sh_y = int((l_sh[1] + r_sh[1]) / 2)
            #mid_wr_y = int((l_wr[1] + r_wr[1]) / 2)

            #cv2.line(frame, (0, mid_sh_y), (w, mid_sh_y), (0, 255, 255), 2)  # sarı: omuz çizgisi
            #cv2.line(frame, (0, mid_wr_y), (w, mid_wr_y), (0, 165, 255), 2)  # turuncu: bar çizgisi

            #dbg = f"BARdrop={bar_drop:+.3f}  (wr={mid_wr_y}, sh={mid_sh_y})"
            #cv2.putText(frame, dbg, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Omuz genişliği (x uzaklığı) normalize ---
            shoulder_span = abs(r_sh[0] - l_sh[0]) / w  # 0..1; protraction artınca küçülür
            item["shoulder_span"] = shoulder_span

            # --- Omuz öne düşme göstergesi (omuz Z - kalça Z) ---
            z_hips = []
            for (u, v) in [l_hip, r_hip]:
                if 0 <= u < w and 0 <= v < h:
                    z_hips.append(float(depth[int(v), int(u)]))
            Zhip = float(np.mean(z_hips)) if z_hips else None
            shoulder_protr = (Zrel - Zhip) if (Zrel is not None and Zhip is not None) else None

            # --- EMA smoothing ---
            if bar_drop is not None:
                ema_bar = bar_drop if ema_bar is None else (ALPHA_BAR * bar_drop + (1 - ALPHA_BAR) * ema_bar)
            else:
                ema_bar = ema_bar  # değişmesin

            if shoulder_protr is not None:
                ema_pro = shoulder_protr if ema_pro is None else (
                            ALPHA_PRO * shoulder_protr + (1 - ALPHA_PRO) * ema_pro)
            else:
                ema_pro = ema_pro

            # item içine smoothed değerleri yaz
            item["bar_drop"] = ema_bar if ema_bar is not None else bar_drop
            item["shoulder_protr"] = ema_pro if ema_pro is not None else shoulder_protr
            dbg = f"BARdrop={item['bar_drop']:+.3f}"

            item = {"elbow_left":L_elb, "elbow_right":R_elb,
                    "shoulder_left":L_shd, "shoulder_right":R_shd, "z":Zrel, "bar_drop": bar_drop, "shoulder_protr": shoulder_protr}

            if render_overlay:
                # iskelet çiz
                drawer.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=style.get_default_pose_landmarks_style())
                # metinler
                def put(y, txt, bad=False):
                    cv2.putText(frame, txt, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0,0,255) if bad else (255,255,255), 2)

                def put_right(y, txt, bad=False, margin=20, font_scale=0.75, thick=2):
                    color = (255, 255, 255) if bad else (0, 255, 255)
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                    x = W - tw - margin
                    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
                    return y + th + 8

                badL = badR = badSL = badSR = badZ = False
                if ranges:
                    r = ranges
                    badL  = not within(L_elb, *r["elbow_left"])
                    badR  = not within(R_elb, *r["elbow_right"])
                    badSL = not within(L_shd, *r["shoulder_left"])
                    badSR = not within(R_shd, *r["shoulder_right"])
                    badZ  = (Zrel is not None) and (not within(Zrel, *r["z"]))
                    issues_counter["elbow_left"]  += int(badL)
                    issues_counter["elbow_right"] += int(badR)
                    issues_counter["shoulder_left"] += int(badSL)
                    issues_counter["shoulder_right"]+= int(badSR)
                    issues_counter["z"] += int(badZ)

                    MARGIN_BAR = 0.02  # %2 tolerans (normalize ölçekte)
                    MARGIN_PRO = 0.02

                    if ranges:
                        r = ranges

                        # ... mevcut badL/badR/badSL/badSR/badZ ...

                        # BAR ve PROTR için margin'li kıyas
                        badBAR = False
                        if item.get("bar_drop") is not None and "bar_drop" in r:
                            lo, hi = r["bar_drop"]
                            val = item["bar_drop"]
                            badBAR = (val < (lo - MARGIN_BAR)) or (val > (hi + MARGIN_BAR))

                        badPROT = False
                        if item.get("shoulder_protr") is not None and "shoulder_protr" in r:
                            lo, hi = r["shoulder_protr"]
                            val = item["shoulder_protr"]
                            badPROT = (val < (lo - MARGIN_PRO)) or (val > (hi + MARGIN_PRO))

                        issues_counter["bar_drop"] += int(badBAR)
                        issues_counter["shoulder_protr"] += int(badPROT)

                        # ekranda da göstermek istersen:
                        #put(y=160, txt=f"BAR: {val:+.3f}", bad=badBAR)
                        #put(y=190, txt=f"PROT: {item['shoulder_protr']:+.3f}", bad=badPROT)
                        right_y = 40
                        right_y = put_right(right_y, f"BAR:  {item['bar_drop']:+.3f}", bad=badBAR)
                        right_y = put_right(right_y, f"PROT: {item['shoulder_protr']:+.3f}", bad=badPROT)

                    # BAR ve OMUZ ÖNE EĞİLME kontrolleri
                    badBAR = not within(bar_drop, *r["bar_drop"])
                    badPROT = not within(shoulder_protr, *r["shoulder_protr"])

                    issues_counter["bar_drop"] += int(badBAR)
                    issues_counter["shoulder_protr"] += int(badPROT)

                put(y=40, txt=f"L-ELB: {L_elb:5.1f}", bad=badL)
                put(y=70, txt=f"R-ELB: {R_elb:5.1f}", bad=badR)
                put(y=100, txt=f"L-SHD: {L_shd:5.1f}", bad=badSL)
                put(y=130, txt=f"R-SHD: {R_shd:5.1f}", bad=badSR)
                if Zrel is not None:
                    put(y=160, txt=f"Zrel : {Zrel:6.3f}", bad=badZ)

                # === FEEDBACK OVERLAY ===
                if ranges:
                    msgs = feedback_text(item, ranges)
                    if msgs:
                        # yarı şeffaf siyah arka plan
                        overlay = frame.copy()
                        box_h = 28 * len(msgs) + 16
                        cv2.rectangle(overlay, (10, 180), (630, 180 + box_h), (0, 0, 0), -1)
                        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

                        # kırmızı uyarı yazısı
                        y0 = 200
                        for i, m in enumerate(msgs):
                            cv2.putText(frame, f"⚠ {m}", (20, y0 + 28*i),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # pose yoksa frame atla
            pass

        metrics.append(item)
        writer.write(frame)
        if ARGS.show:
            cv2.imshow("pose+depth", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')): break

    cap.release(); writer.release()
    if ARGS.show: cv2.destroyAllWindows()
    return metrics, issues_counter

# ============ REFERANS ÜRET (ref) ============
def build_reference():
    metrics, _ = extract_metrics(ARGS.src, render_overlay=True, ranges=None)

    # ---- ORTA DİLİM FİLTRESİ (örn. %20–%80) ----
    LO_FRAC = 0.20  # alt sınır (değiştirebilirsin)
    HI_FRAC = 0.80  # üst sınır (değiştirebilirsin)

    n = len(metrics)
    start = int(n * LO_FRAC)
    end = int(n * HI_FRAC)
    metrics = metrics[start:end]
    print(f"[REF] Orta dilim seçildi: {start}:{end} / {n} kare")

    # Her metrik için 15-85 yüzdelik aralığı
    def series(key): return [m[key] for m in metrics if m[key] is not None]
    ref = {
        "elbow_left"    : percentile_range(series("elbow_left")),
        "elbow_right"   : percentile_range(series("elbow_right")),
        "shoulder_left" : percentile_range(series("shoulder_left")),
        "shoulder_right": percentile_range(series("shoulder_right")),
        # Zrel: animasyon/videodan bağımsızlaştırmak için normalize edelim (frame-içi z-score)
        # Ama basit tutuyoruz: ref videodaki "raw" yüzdelik aralığı alıyoruz.
        "z"             : percentile_range(series("z")),
        "meta": {"lo":15, "hi":85, "note":"MiDaS relative depth; omuz ortalaması"},
        "bar_drop": percentile_range([m["bar_drop"] for m in metrics if m.get("bar_drop") is not None]),
        "shoulder_protr": percentile_range(
            [m["shoulder_protr"] for m in metrics if m.get("shoulder_protr") is not None]),"shoulder_span": percentile_range([m["shoulder_span"] for m in metrics if m.get("shoulder_span") is not None]),


    }
    with open(ARGS.refjson, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
    print("[REF] yazıldı ->", ARGS.refjson)

    if ARGS.save_metrics:
        with open(ARGS.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[REF] ham metrikler ->", ARGS.save_metrics)

# ============ DEĞERLENDİR (eval) ============
def evaluate():
    if not Path(ARGS.refjson).exists():
        raise SystemExit(f"[HATA] referans yok: {ARGS.refjson} (önce --mode ref çalıştır)")
    with open(ARGS.refjson, "r", encoding="utf-8") as f:
        ref = json.load(f)

    metrics, issues_counter = extract_metrics(ARGS.src, render_overlay=True, ranges=ref)
    totals = {k:int(v) for k,v in issues_counter.items()}
    bad_frames = sum(1 for m in metrics if None in m.values())  # pose çıkmayan kare sayısı
    summary = {
        "video": ARGS.src,
        "frames": len(metrics),
        "pose_missing_frames": bad_frames,
        "violations_count": totals,
        "ref_used": ARGS.refjson
    }
    report_path = Path(ARGS.out).with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[EVAL] rapor ->", report_path)
    if ARGS.save_metrics:
        with open(ARGS.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[EVAL] ham metrikler ->", ARGS.save_metrics)

# ============ ÇALIŞTIR ============
if __name__ == "__main__":
    if ARGS.mode == "ref":
        build_reference()
    else:
        evaluate()
