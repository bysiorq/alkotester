#!/usr/bin/env python3
"""
Demo krok-po-kroku zgodne z pipeline alkotestera:
Haar (największa twarz) -> resize 240x240 -> ORB keypoints/deskryptory.

Domyślnie:
    python3 demo_step_by_step.py
pobiera JEDEN kadr z kamery (Picamera2 / GStreamer / /dev/video0),
przetwarza go i zapisuje kolejne etapy do katalogu demo_output/:

    step1_original_bgr.jpg
    step2_gray.jpg
    step3_haar_box.jpg
    step4_face_240_gray.jpg
    step5_orb_keypoints.jpg

Zero GUI, gotowe do wklejenia jako ilustracja działania algorytmu.
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# -------------------------------
# Znalezienie klasyfikatora Haar
# -------------------------------
def find_haar():
    fname = "haarcascade_frontalface_default.xml"
    candidates = []

    # Jeśli dostępne: ścieżka z cv2.data
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(cv2.data.haarcascades)

    # Typowe lokalizacje na RPi / Debianie
    candidates += [
        "/usr/share/opencv4/haarcascades/",
        "/usr/share/opencv/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
        "./",
    ]

    for base in candidates:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            print(f"[INFO] Haar cascade: {path}")
            return path

    print("[ERROR] Nie znaleziono haarcascade_frontalface_default.xml.")
    print("Skopiuj plik do katalogu projektu lub doinstaluj pakiet z kaskadami.")
    sys.exit(1)

# -------------------------------
# Pobranie jednej klatki z kamery
# Kluczowa poprawka: format XRGB8888 + BGRA->BGR
# -------------------------------
def capture_one_frame(size=(1920, 1080), fps=30):
    # 1) Picamera2 (CSI) – rekomendowany sposób
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()

        # Uwaga: XRGB8888 daje 4 kanały, stabilne z OpenCV
        config = picam2.create_still_configuration(
            main={"size": size, "format": "XRGB8888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.3)  # krótki rozbieg
        frame = picam2.capture_array()  # BGRA (XRGB8888)
        picam2.stop()

        if frame is None:
            raise RuntimeError("Picamera2 zwróciła pusty kadr.")

        # KONWERSJA KLUCZOWA:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        print("[INFO] Klatka z kamery (Picamera2, XRGB8888 -> BGR).")
        return bgr
    except Exception as e:
        print(f"[WARN] Picamera2 nie działa poprawnie: {e}")

    # 2) GStreamer + libcamerasrc (fallback)
    gst = (
        f"libcamerasrc ! "
        f"video/x-raw,width={size[0]},height={size[1]},framerate={fps}/1 ! "
        f"videoconvert ! appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print("[INFO] Klatka z kamery (GStreamer libcamerasrc).")
            return frame
        print("[WARN] GStreamer uruchomiony, ale brak poprawnej klatki.")

    # 3) /dev/video0 (USB / bridge)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print("[INFO] Klatka z kamery (/dev/video0).")
            return frame

    raise RuntimeError("Nie udało się pobrać klatki z kamery (Picamera2/GStreamer/V4L2).")

# -------------------------------
# Pipeline krok po kroku -> pliki
# -------------------------------
def run_pipeline_and_save(bgr, out_dir="demo_output"):
    os.makedirs(out_dir, exist_ok=True)

    # Krok 1: oryginalny obraz
    p1 = os.path.join(out_dir, "step1_original_bgr.jpg")
    cv2.imwrite(p1, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] Zapisano: {p1}")

    # Krok 2: skala szarości
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p2 = os.path.join(out_dir, "step2_gray.jpg")
    cv2.imwrite(p2, gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] Zapisano: {p2}")

    # Krok 3: detekcja Haar
    cascade = cv2.CascadeClassifier(find_haar())
    if cascade.empty():
        print("[ERROR] Nie udało się załadować klasyfikatora Haar.")
        sys.exit(1)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        print("[WARN] Nie wykryto twarzy. Te pliki i tak możesz użyć jako ilustrację wejścia i konwersji.")
        return

    # największa twarz – jak w FaceDB
    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

    bgr_box = bgr.copy()
    cv2.rectangle(bgr_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    p3 = os.path.join(out_dir, "step3_haar_box.jpg")
    cv2.imwrite(p3, bgr_box, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] Zapisano: {p3}")

    # Krok 4: wycięta twarz 240x240 (gray)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (240, 240), interpolation=cv2.INTER_LINEAR)
    p4 = os.path.join(out_dir, "step4_face_240_gray.jpg")
    cv2.imwrite(p4, roi_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] Zapisano: {p4}")

    # Krok 5: ORB – punkty kluczowe i deskryptory
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(roi_gray, None)

    vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(
        roi_gray,
        keypoints,
        vis,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    p5 = os.path.join(out_dir, "step5_orb_keypoints.jpg")
    cv2.imwrite(p5, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] Zapisano: {p5}")

    if descriptors is not None and len(keypoints) > 0:
        print(f"[INFO] Liczba punktów ORB: {len(keypoints)}")
        print(f"[INFO] Kształt deskryptorów: {descriptors.shape} (N x 32)")
    else:
        print("[WARN] Brak deskryptorów ORB – sprawdź ostrość/oświetlenie.")

# -------------------------------
# main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Demo Haar+ORB: kamera domyślnie, albo --image plik. Zapisuje etapy do demo_output/."
    )
    ap.add_argument("--image", help="Ścieżka do obrazu (opcjonalnie).")
    args = ap.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"[ERROR] Nie udało się wczytać obrazu: {args.image}")
            sys.exit(1)
    else:
        try:
            bgr = capture_one_frame()
        except Exception as e:
            print(f"[ERROR] Nie udało się pobrać klatki z kamery: {e}")
            print("Możesz awaryjnie użyć: libcamera-jpeg -o selfie.jpg i uruchomić z --image selfie.jpg")
            sys.exit(1)

    run_pipeline_and_save(bgr)

if __name__ == "__main__":
    main()
