#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wsadowe demo detekcji/deskrypcji twarzy z polskimi (ASCII/freetype) adnotacjami:
- YuNet (wieloprogowo / wieloskalowo) + fallback SSD i Haar
- Haar + ORB (czytelne wizualizacje: FAST, siatka, surowe ORB, rozrzedzone ORB, orientacje, latki)

Dla kazdego .jpg w katalogu wejsciowym tworzy podkatalog 1/, 2/, ... z plikami:
  00_meta_zrodlo_roi.jpg
  01_wejscie_bgr.jpg
  02_skala_szarosci.jpg
  03_yunet_proba_XX_scale_S_score_T.jpg (dla kazdej proby)
  04_yunet_najlepsza.jpg  (albo *_brak_detekcji.jpg)
  05_fallback_ssd.jpg
  06_fallback_haar.jpg
  07_twarz_240_bgr.jpg
  08_twarz_240_gray.jpg
  09_fast_rogi.jpg
  10_siatka_8x8.jpg
  11_orb_surowe.jpg
  12_orb_po_siatce.jpg
  13_orb_orientacje.jpg
  14_orb_latki_31x31.jpg
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np

# --------------------- utils: katalogi i zapis ---------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def savejpg(path, img, q=95):
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])

# --------------------- obsluga czcionek ---------------------

# Mapowanie polskich znakow -> ASCII (fallback)
_PL_MAP = str.maketrans({
    "ą":"a","ć":"c","ę":"e","ł":"l","ń":"n","ó":"o","ś":"s","ź":"z","ż":"z",
    "Ą":"A","Ć":"C","Ę":"E","Ł":"L","Ń":"N","Ó":"O","Ś":"S","Ź":"Z","Ż":"Z"
})

def ascii_pl(s: str) -> str:
    try:
        return s.translate(_PL_MAP)
    except Exception:
        return s

# Proba konfiguracji cv2.freetype (Arial / inny TTF)
_FT2 = None
_USE_FT = False

def _init_freetype():
    global _FT2, _USE_FT
    try:
        # cv2.freetype jest w opencv-contrib
        import cv2.freetype as ft
    except Exception:
        _FT2 = None
        _USE_FT = False
        return

    # typowe lokalizacje fontow TrueType
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    font_path = None
    for p in candidates:
        if os.path.exists(p):
            font_path = p
            break

    if font_path is None:
        _FT2 = None
        _USE_FT = False
        return

    try:
        _FT2 = ft.createFreeType2()
        _FT2.loadFontData(fontFileName=font_path, id=0)
        _USE_FT = True
        print(f"[INFO] Uzywam TrueType fontu: {font_path}")
    except Exception as e:
        print(f"[WARN] Nie udalo sie zainicjowac freetype: {e}")
        _FT2 = None
        _USE_FT = False

_init_freetype()

def put_tag(img, text, org=(10, 25), color=(0, 255, 255),
            scale=0.7, thickness=2):
    """
    Wypisuje podpis:
      - TrueType (Arial / DejaVu) przez cv2.freetype, jesli dostepne,
      - w przeciwnym razie zwykle cv2.putText z transliteracja na ASCII.
    """
    if _USE_FT and _FT2 is not None:
        txt = text  # pelne polskie znaki
        font_height = int(22 * scale)
        try:
            _FT2.putText(
                img, txt, org,
                fontHeight=font_height,
                color=color,
                thickness=thickness,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
            return
        except Exception:
            # jesli freetype padnie w runtime, fallback na zwykle putText
            pass

    # fallback – bez ogonkow, Hershey
    txt = ascii_pl(text)
    cv2.putText(
        img, txt, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )

# --------------------- inne utils ---------------------

def find_haar():
    fname = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(cv2.data.haarcascades)
    candidates += [
        "/usr/share/opencv4/haarcascades/",
        "/usr/share/opencv/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
        "./",
    ]
    for base in candidates:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    return fname

def draw_landmarks_pl(bgr, lm, color=(0, 255, 0)):
    # nazwy bez ogonkow (ASCII) – zadziała wszedzie
    names = [
        "Lewe oko",
        "Prawe oko",
        "Nos",
        "Lewy kacik ust",
        "Prawy kacik ust",
    ]
    for i in range(5):
        x = int(lm[2 * i])
        y = int(lm[2 * i + 1])
        cv2.circle(bgr, (x, y), 2, color, -1, cv2.LINE_AA)
        put_tag(bgr, names[i], (x + 3, y - 3), color, scale=0.4, thickness=1)

def draw_grid_overlay(gray240, grid=8):
    vis = cv2.cvtColor(gray240, cv2.COLOR_GRAY2BGR)
    step = gray240.shape[1] // grid
    for i in range(1, grid):
        x = i * step
        cv2.line(vis, (x, 0), (x, 239), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(vis, (0, x), (239, x), (60, 60, 60), 1, cv2.LINE_AA)
    put_tag(vis, f"Siatka {grid}x{grid}", (10, 22), (200, 200, 200), scale=0.6, thickness=1)
    return vis

def limit_keypoints_grid(kps, grid=8, size=240, per_cell=3):
    cells = [[[] for _ in range(grid)] for _ in range(grid)]
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cx = max(0, min(grid - 1, x * grid // size))
        cy = max(0, min(grid - 1, y * grid // size))
        if len(cells[cy][cx]) < per_cell:
            cells[cy][cx].append(kp)
    out = []
    for row in cells:
        for cell in row:
            out.extend(cell)
    return out

def extract_patch(gray, x, y, size=31):
    r = size // 2
    h, w = gray.shape[:2]
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
    patch = np.zeros((size, size), dtype=gray.dtype)
    crop = gray[y0:y1, x0:x1]
    patch[0:crop.shape[0], 0:crop.shape[1]] = crop
    return patch

def make_patches_montage(gray240, kps, count=20, size=31, cols=10):
    chosen = kps[:count]
    rows = (len(chosen) + cols - 1) // cols
    pad = 2
    H = rows * (size + pad) + pad
    W = cols * (size + pad) + pad
    mono = np.full((H, W), 30, dtype=np.uint8)
    for i, kp in enumerate(chosen):
        r = i // cols
        c = i % cols
        px = int(round(kp.pt[0]))
        py = int(round(kp.pt[1]))
        p = extract_patch(gray240, px, py, size=size)
        y0 = pad + r * (size + pad)
        x0 = pad + c * (size + pad)
        mono[y0:y0 + size, x0:x0 + size] = p
    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

# --------------------- detektory ---------------------

def detect_yunet_multiscale(
    bgr, model_path,
    score_list=(0.85, 0.60, 0.40),
    scale_list=(1.0, 1.5, 2.0),
    nms=0.30,
    topk=5000
):
    """
    Zwraca:
      best: dict lub None:
            {'best_score': float, 'scale': s, 'score_th': t, 'dets': [...]}
      tried: lista prob (dla wizualizacji)
    """
    if not (hasattr(cv2, "FaceDetectorYN_create") and os.path.exists(model_path)):
        return None, []

    h0, w0 = bgr.shape[:2]
    det = cv2.FaceDetectorYN_create(
        model_path, "", (w0, h0),
        float(score_list[0]), float(nms), int(topk)
    )
    tried = []
    best = None

    for t in score_list:
        det.setScoreThreshold(float(t))
        for s in scale_list:
            if s == 1.0:
                img = bgr
            else:
                img = cv2.resize(
                    bgr,
                    (int(w0 * s), int(h0 * s)),
                    interpolation=cv2.INTER_LINEAR
                )
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            try:
                _, faces = det.detect(img)
            except Exception:
                faces = None

            dets = []
            if faces is not None and len(faces) > 0:
                for f in faces:
                    x, y, wf, hf = f[:4]
                    score = float(f[14]) if len(f) >= 15 else float(t)
                    lm = f[4:14].copy()
                    # przeskalowanie do oryginalu
                    x, y, wf, hf = x / s, y / s, wf / s, hf / s
                    for i in range(5):
                        lm[2 * i] = lm[2 * i] / s
                        lm[2 * i + 1] = lm[2 * i + 1] / s
                    dets.append({
                        "box": (
                            int(round(x)),
                            int(round(y)),
                            int(round(wf)),
                            int(round(hf)),
                        ),
                        "score": score,
                        "lm": lm.tolist(),
                    })

            tried.append({
                "scale": float(s),
                "score_th": float(t),
                "dets": dets,
            })

            for d in dets:
                if (best is None) or (d["score"] > best["best_score"]):
                    best = {
                        "best_score": d["score"],
                        "scale": float(s),
                        "score_th": float(t),
                        "dets": dets,
                    }

    return best, tried

def detect_ssd(bgr, proto, model, conf_th=0.60):
    if not (os.path.exists(proto) and os.path.exists(model)):
        return []
    try:
        net = cv2.dnn.readNetFromCaffe(proto, model)
    except Exception:
        return []
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        bgr, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    det = net.forward()
    out = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf >= conf_th:
            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                out.append({
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "score": conf,
                })
    return out

def detect_haar(gray):
    cascade = cv2.CascadeClassifier(find_haar())
    if cascade.empty():
        return []
    faces = cascade.detectMultiScale(gray, 1.2, 5)
    return [
        {"box": (int(x), int(y), int(w), int(h)), "score": 1.0}
        for (x, y, w, h) in faces
    ]

# --------------------- core per-image ---------------------

def process_one_image(img_path, out_dir, args):
    ensure_dir(out_dir)

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[ERROR] Nie wczytano: {img_path}")
        return

    # 01-02: wejscie + gray (z lekkim CLAHE dla stabilnosci ORB)
    savejpg(os.path.join(out_dir, "01_wejscie_bgr.jpg"), bgr, 95)
    gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray_raw)
    savejpg(os.path.join(out_dir, "02_skala_szarosci.jpg"), gray, 95)

    # 03: YuNet – kazda proba osobno
    best, tried = detect_yunet_multiscale(
        bgr,
        args.yunet,
        score_list=args.scores,
        scale_list=args.scales,
        nms=0.30,
        topk=5000,
    )
    for idx, attempt in enumerate(tried, start=1):
        sc = attempt["scale"]
        thr = attempt["score_th"]
        dets = attempt["dets"]
        vis = bgr.copy()
        tag = f"YuNet s={sc:.2f}, prog>={thr:.2f}"
        put_tag(vis, tag, (10, 25), (0, 255, 0), scale=0.7, thickness=2)

        if dets:
            for d in dets:
                (x, y, w, h) = d["box"]
                s = d["score"]
                cv2.rectangle(
                    vis, (x, y), (x + w, y + h),
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                if "lm" in d and d["lm"]:
                    draw_landmarks_pl(vis, d["lm"], (0, 255, 0))
                # krótka etykieta przy ramce
                put_tag(
                    vis,
                    f"p={s:.2f}",
                    (x + 5, max(15, y - 8)),
                    (0, 255, 0),
                    scale=0.6,
                    thickness=1,
                )
        else:
            put_tag(
                vis,
                "Brak twarzy",
                (10, 55),
                (0, 255, 255),
                scale=0.7,
                thickness=2,
            )

        savejpg(
            os.path.join(
                out_dir,
                f"03_yunet_proba_{idx:02d}_scale_{sc:.2f}_score_{thr:.2f}.jpg",
            ),
            vis,
            95,
        )

    # 04: najlepsza YuNet (albo brak)
    vis_best = bgr.copy()
    yunet_dets = []
    if best is not None and best["dets"]:
        yunet_dets = best["dets"]
        for d in yunet_dets:
            (x, y, w, h) = d["box"]
            s = d["score"]
            cv2.rectangle(
                vis_best, (x, y), (x + w, y + h),
                (0, 200, 0), 2, cv2.LINE_AA
            )
            if "lm" in d and d["lm"]:
                draw_landmarks_pl(vis_best, d["lm"], (0, 200, 0))
            put_tag(
                vis_best,
                f"p={s:.2f}",
                (x + 5, max(15, y - 8)),
                (0, 200, 0),
                scale=0.6,
                thickness=1,
            )
        put_tag(
            vis_best,
            f"YuNet najlepszy (s={best['scale']:.2f}, prog>={best['score_th']:.2f})",
            (10, 25),
            (0, 200, 0),
            scale=0.7,
            thickness=2,
        )
        savejpg(os.path.join(out_dir, "04_yunet_najlepsza.jpg"), vis_best, 95)
    else:
        put_tag(
            vis_best,
            "YuNet: brak detekcji",
            (10, 25),
            (0, 255, 255),
            scale=0.8,
            thickness=2,
        )
        savejpg(
            os.path.join(out_dir, "04_yunet_najlepsza_brak_detekcji.jpg"),
            vis_best,
            95,
        )

    # 05: fallback SSD (tylko jesli YuNet nic nie wybral)
    ssd_dets = []
    if not yunet_dets:
        ssd_dets = detect_ssd(
            bgr,
            args.ssd_proto,
            args.ssd_model,
            conf_th=0.60,
        )
    vis_ssd = bgr.copy()
    if ssd_dets:
        for d in ssd_dets:
            (x, y, w, h) = d["box"]
            cv2.rectangle(
                vis_ssd, (x, y), (x + w, y + h),
                (255, 0, 0), 2, cv2.LINE_AA
            )
            put_tag(
                vis_ssd,
                f"p={d['score']:.2f}",
                (x + 5, max(15, y - 8)),
                (255, 0, 0),
                scale=0.6,
                thickness=1,
            )
        put_tag(
            vis_ssd,
            "SSD fallback",
            (10, 25),
            (255, 0, 0),
            scale=0.7,
            thickness=2,
        )
    else:
        put_tag(
            vis_ssd,
            "SSD: brak detekcji",
            (10, 25),
            (0, 255, 255),
            scale=0.8,
            thickness=2,
        )
    savejpg(os.path.join(out_dir, "05_fallback_ssd.jpg"), vis_ssd, 95)

    # 06: fallback Haar (tylko jesli YuNet i SSD puste)
    haar_dets = []
    if not yunet_dets and not ssd_dets:
        haar_dets = detect_haar(gray)
    vis_haar = bgr.copy()
    if haar_dets:
        for d in haar_dets:
            (x, y, w, h) = d["box"]
            cv2.rectangle(
                vis_haar, (x, y), (x + w, y + h),
                (0, 255, 255), 2, cv2.LINE_AA
            )
        put_tag(
            vis_haar,
            "Haar fallback",
            (10, 25),
            (0, 255, 255),
            scale=0.7,
            thickness=2,
        )
    else:
        put_tag(
            vis_haar,
            "Haar: brak detekcji",
            (10, 25),
            (0, 0, 255),
            scale=0.8,
            thickness=2,
        )
    savejpg(os.path.join(out_dir, "06_fallback_haar.jpg"), vis_haar, 95)

    # 07-14: ROI (YuNet > SSD > Haar, inaczej srodek)
    chosen = None
    for arr in (yunet_dets, ssd_dets, haar_dets):
        if arr:
            chosen = max(arr, key=lambda d: d["box"][2] * d["box"][3])
            break

    if chosen is not None:
        x, y, w, h = chosen["box"]
        face_bgr = bgr[y:y + h, x:x + w].copy()
        roi_source = "detekcja (YuNet/SSD/Haar)"
    else:
        H, W = bgr.shape[:2]
        sz = min(H, W, 240)
        cx, cy = W // 2, H // 2
        x0 = max(0, cx - sz // 2)
        y0 = max(0, cy - sz // 2)
        face_bgr = bgr[y0:y0 + sz, x0:x0 + sz].copy()
        roi_source = "crop srodek"

    face_bgr = cv2.resize(face_bgr, (240, 240), interpolation=cv2.INTER_LINEAR)
    savejpg(os.path.join(out_dir, "07_twarz_240_bgr.jpg"), face_bgr, 95)

    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    savejpg(os.path.join(out_dir, "08_twarz_240_gray.jpg"), face_gray, 95)

    # 09: FAST – co widzi detektor rogow
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kps_fast = fast.detect(face_gray, None) or []
    vis_fast = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(
        face_gray, kps_fast, vis_fast,
        color=(0, 255, 255)
    )
    put_tag(
        vis_fast,
        f"Rogi FAST: {len(kps_fast)}",
        (10, 22),
        (0, 255, 255),
        scale=0.7,
        thickness=2,
    )
    savejpg(os.path.join(out_dir, "09_fast_rogi.jpg"), vis_fast, 95)

    # 10: siatka 8x8
    grid_vis = draw_grid_overlay(face_gray, grid=8)
    savejpg(os.path.join(out_dir, "10_siatka_8x8.jpg"), grid_vis, 95)

    # 11: ORB surowe
    orb = cv2.ORB_create(nfeatures=1000)
    kps_raw, des = orb.detectAndCompute(face_gray, None)
    if kps_raw is None:
        kps_raw = []
    vis_raw = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(
        face_gray, kps_raw, vis_raw,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    put_tag(
        vis_raw,
        f"ORB surowe: {len(kps_raw)}",
        (10, 22),
        (200, 200, 255),
        scale=0.7,
        thickness=2,
    )
    savejpg(os.path.join(out_dir, "11_orb_surowe.jpg"), vis_raw, 95)

    # 12: ORB – rozrzedzenie po siatce 8x8
    per_cell = max(1, min(5, int(round(args.orb_points / 64.0))))
    kps_tamed = limit_keypoints_grid(
        kps_raw,
        grid=8,
        size=240,
        per_cell=per_cell
    )
    vis_tamed = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(
        face_gray, kps_tamed, vis_tamed,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    put_tag(
        vis_tamed,
        f"ORB po siatce: {len(kps_tamed)} (8x8, {per_cell}/kom)",
        (10, 22),
        (0, 255, 0),
        scale=0.6,
        thickness=2,
    )
    savejpg(os.path.join(out_dir, "12_orb_po_siatce.jpg"), vis_tamed, 95)

    # 13: ORB – orientacje (strzalki)
    vis_orient = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    for kp in kps_tamed:
        xk = int(round(kp.pt[0]))
        yk = int(round(kp.pt[1]))
        ang_deg = kp.angle if kp.angle is not None else 0.0
        ang = np.deg2rad(ang_deg)
        dx = int(8 * np.cos(ang))
        dy = int(8 * np.sin(ang))
        cv2.circle(
            vis_orient,
            (xk, yk),
            2,
            (0, 200, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            vis_orient,
            (xk, yk),
            (xk + dx, yk + dy),
            (0, 200, 255),
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
    put_tag(
        vis_orient,
        "ORB orientacje",
        (10, 22),
        (0, 200, 255),
        scale=0.7,
        thickness=2,
    )
    savejpg(os.path.join(out_dir, "13_orb_orientacje.jpg"), vis_orient, 95)

    # 14: ORB – latki 31x31 (top wg response)
    kps_sorted = sorted(
        kps_tamed,
        key=lambda k: k.response,
        reverse=True
    )
    montage = make_patches_montage(
        face_gray,
        kps_sorted,
        count=min(20, len(kps_sorted)),
        size=31,
        cols=10,
    )
    put_tag(
        montage,
        "Latki ORB 31x31 - top",
        (10, 22),
        (255, 255, 255),
        scale=0.7,
        thickness=2,
    )
    savejpg(os.path.join(out_dir, "14_orb_latki_31x31.jpg"), montage, 95)

    # 00: metka zrodla ROI
    roi_info = cv2.cvtColor(
        np.full((30, 240), 20, dtype=np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    put_tag(
        roi_info,
        f"Zrodlo ROI: {roi_source}",
        (10, 22),
        (180, 180, 180),
        scale=0.6,
        thickness=1,
    )
    savejpg(os.path.join(out_dir, "00_meta_zrodlo_roi.jpg"), roi_info, 95)

# --------------------- batch main ---------------------

def parse_floats_csv(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return tuple(out)

def main():
    ap = argparse.ArgumentParser(
        description="Wsadowe YuNet + Haar/ORB demo (czytelne polskie adnotacje)."
    )
    ap.add_argument(
        "--in_dir",
        default="demo-images",
        help="Katalog ze zdjeciami .jpg",
    )
    ap.add_argument(
        "--yunet",
        required=True,
        help="Sciezka do face_detection_yunet_2023mar.onnx",
    )
    ap.add_argument(
        "--ssd_proto",
        default="models/deploy.prototxt",
        help="(opcjonalnie) deploy.prototxt",
    )
    ap.add_argument(
        "--ssd_model",
        default="models/res10_300x300_ssd_iter_140000.caffemodel",
        help="(opcjonalnie) caffemodel",
    )
    ap.add_argument(
        "--scores",
        default="0.85,0.60,0.40",
        help="Progi YuNet CSV, np. 0.85,0.60,0.40",
    )
    ap.add_argument(
        "--scales",
        default="1.0,1.5,2.0",
        help="Skale YuNet CSV, np. 1.0,1.5,2.0",
    )
    ap.add_argument(
        "--orb_points",
        type=int,
        default=120,
        help="Docelowa liczba punktow ORB po rozrzedzeniu (ok. 120)",
    )
    args = ap.parse_args()

    args.scores = parse_floats_csv(args.scores)
    args.scales = parse_floats_csv(args.scales)

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.jpg")))
    if not files:
        print(f"[ERROR] Brak plikow .jpg w: {args.in_dir}")
        sys.exit(1)

    for idx, img_path in enumerate(files, start=1):
        out_dir = os.path.join(args.in_dir, str(idx))
        print(f"[INFO] {idx}: {os.path.basename(img_path)}  ->  {out_dir}")
        process_one_image(img_path, out_dir, args)

    print("[OK] Gotowe.")

if __name__ == "__main__":
    main()
