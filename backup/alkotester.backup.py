#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alkotester – system na Raspberry Pi 4 (8 GB) zgodny ze schematem blokowym
--------------------------------------------------------------------------
Funkcje:
- Podgląd z kamery Raspberry Pi Camera v2 w pełnym oknie (720x1280, pion)
- Obsługa dotyku (DSI Touch Display 2 – traktowany jak mysz)
- Odczyt MQ-3 przez MCP3008 na SPI0 (spidev), kalibracja tła, estymacja promili [‰]
- Ekranowy stan aplikacji zgodny z blokowym scenariuszem: rozpoznanie twarzy →
  kalibracja → pomiar → decyzja: otwarcie przejścia / ponów pomiar / odmowa
- Rozpoznawanie twarzy: lekki, offline’owy rozpoznawacz (OpenCV Haar + ORB),
  baza zdjęć w folderach per-pracownik, wariant A/B (nowy / istniejący),
  zastępowanie najstarszego zdjęcia oraz „trening” (reindeksacja deskryptorów)
- PIN fallback: gdy kalibracja nieudana / brak rozpoznania – wejście po PIN
- Logi CSV oraz sygnał GPIO na przekaźnik „otwarcie przejścia”

Zależności (Raspberry Pi OS Bookworm):
  sudo apt update && sudo apt install -y \
    python3-picamera2 python3-opencv python3-pyside6 python3-numpy \
    python3-spidev python3-rpi.gpio

Uruchomienie:
  1) Aktywuj SPI:  sudo raspi-config  → Interface Options → SPI → Enable
  2) Przetestuj kamerę:  libcamera-hello  (powinno pokazać podgląd)
  3) Uruchom appkę:  python3 alkotester.py

Połączenia MCP3008 (SPI0, CE0):
  MCP3008 VDD → 3V3, VREF → 3V3, AGND → GND, DGND → GND
  MCP3008 CLK → GPIO11 (SCLK), DOUT → GPIO9 (MISO), DIN → GPIO10 (MOSI), CS/SHDN → GPIO8 (CE0)
  MQ-3 (analog) → MCP3008 CH0 (domyślnie), drugi pin MQ-3 do GND, zasilanie 5V (lub 3V3 – wg modułu) i masa wspólna.

Uwaga dot. metrologii:
  MQ-3 nie daje bezpośrednio promili. Ten kod robi kalibrację „tła” (czyste powietrze)
  i liniową estymację [‰] względem współczynnika skalującego. Do zastosowań
  profesjonalnych wymagane jest wzorcowanie i algorytm przeliczeniowy zgodnie
  z kartą katalogową i warunkami pracy (temperatura, wilgotność, przepływ, itp.).

Struktura danych:
  data/employees.json          – lista pracowników + PIN
  data/faces/<emp_id>/*.jpg    – zdjęcia w bazie
  data/index/<emp_id>.npz      – zindeksowane deskryptory ORB
  logs/events.csv, logs/measurements.csv

Zgodność ze schematem blokowym (teksty UI, warianty A/B, PIN fallback, upload flag):
  Implementujemy komunikaty i przebieg (np. „Cześć Andrzej”, „Podejdź bliżej”,
  „Nie wykryto twarzy…”, „Wprowadź PIN”, „A dmuchnij no…”, „Proszę czekać…”,
  test po aktualizacji, raporty/otwarcia). „Trening AI” = przebudowa indeksu ORB.

Autor: GPT-5 Thinking (2025)
"""

import os
import sys
import cv2
import json
import time
import glob
import math
import queue
import shutil
import signal
import threading
import numpy as np
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
# Zmiana z PySide6 na PyQt5 (łatwiej zainstalować z APT na RPi)
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

# Kamera (Picamera2)
from picamera2 import Picamera2

# SPI + GPIO
import spidev
import RPi.GPIO as GPIO


############################
# KONFIGURACJA APLIKACJI   #
############################

CONFIG = {
    # Ekran
    "screen_width": 720,
    "screen_height": 1280,   # pion (portret)
    "fullscreen": True,
    "rotate_180": False,     # jeśli ekran obrócony (np. przewody u góry)

    # Kamera
    "camera_main_size": (1280, 720),   # stream z kamery – dopasujemy do portretu
    "camera_fps": 30,

    # SPI / MCP3008 / MQ-3
    "spi_bus": 0,
    "spi_device": 0,         # CE0
    "mq3_channel": 0,
    "baseline_samples": 150, # ile próbek do ustalenia tła na starcie
    "promille_scale": 220.0, # współczynnik liniowej skali [surowe - tło] → [‰]
    "measure_seconds": 3.0,  # czas rzeczywistego pomiaru dmuchania

    # Progi decyzyjne [‰]
    "threshold_pass": 0.00,   # przejście auto poniżej/na tym poziomie
    "threshold_retry": 0.20,  # powyżej/pass do retry
    "threshold_deny": 0.50,   # powyżej/na tym – odmowa

    # Rozpoznawanie twarzy
    "face_min_size": 120,     # minimalny wymiar twarzy (px) do kalibracji
    "recognition_conf_ok": 55.0,   # % – akceptacja rozpoznania (górny próg)
    "recognition_conf_low": 20.0,  # % – poniżej → traktuj jako brak rozpoznania

    # GPIO przekaźnik „przejście otwarte” (aktywny stan wysoki)
    "gate_gpio": 18,
    "gate_pulse_sec": 5.0,

    # Ścieżki
    "data_dir": "data",
    "faces_dir": "data/faces",
    "index_dir": "data/index",
    "employees_json": "data/employees.json",
    "logs_dir": "logs",

    # Inne
    "debug": False,
}


############################
# NARZĘDZIA: pliki / logi  #
############################

def ensure_dirs():
    for p in [CONFIG["data_dir"], CONFIG["faces_dir"], CONFIG["index_dir"], CONFIG["logs_dir"]]:
        os.makedirs(p, exist_ok=True)
    if not os.path.exists(CONFIG["employees_json"]):
        with open(CONFIG["employees_json"], "w", encoding="utf-8") as f:
            json.dump({"employees": []}, f, ensure_ascii=False, indent=2)


def now_str():
    return datetime.now().strftime("%H:%M %d.%m.%Y")


def log_csv(path, header, row_values):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(";".join(header) + "\n")
        f.write(";".join(map(str, row_values)) + "\n")


############################
# SPI/MCP3008 + MQ-3       #
############################

class MCP3008:
    def __init__(self, bus=0, device=0, max_speed_hz=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = 0

    def read_channel(self, ch: int) -> int:
        # 10-bit adc – standardowa sekwencja: [1, (8+ch)<<4, 0]
        if ch < 0 or ch > 7:
            raise ValueError("MCP3008 channel 0..7")
        r = self.spi.xfer2([1, (8 | ch) << 4, 0])
        value = ((r[1] & 3) << 8) | r[2]
        return value  # 0..1023

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass


class MQ3Sensor:
    def __init__(self, adc: MCP3008, channel: int, baseline_samples: int, promille_scale: float):
        self.adc = adc
        self.channel = channel
        self.baseline_samples = baseline_samples
        self.promille_scale = promille_scale
        self.baseline = None

    def calibrate_baseline(self):
        samples = []
        for _ in range(self.baseline_samples):
            samples.append(self.adc.read_channel(self.channel))
            time.sleep(0.01)
        self.baseline = float(np.median(samples))
        return self.baseline

    def read_raw(self):
        return self.adc.read_channel(self.channel)

    def read_promille(self, duration_sec: float = 1.0):
        t0 = time.time()
        vals = []
        while (time.time() - t0) < duration_sec:
            vals.append(self.read_raw())
            time.sleep(0.005)
        v = float(np.mean(vals)) if vals else float(self.read_raw())
        if self.baseline is None:
            self.baseline = v
        delta = max(0.0, v - self.baseline)
        promille = delta / max(1e-6, self.promille_scale)
        return promille, v


############################
# Rozpoznawanie twarzy     #
############################

class FaceDB:
    """
    Lekka baza: pracownicy + PIN + obrazy + indeksy ORB.
    Folder per pracownik w data/faces/<emp_id> z *.jpg.
    Indeks: data/index/<emp_id>.npz – zdeskryptory ORB dla zdjęć.
    """
    def __init__(self, faces_dir, index_dir, employees_json):
        self.faces_dir = faces_dir
        self.index_dir = index_dir
        self.employees_json = employees_json
        self._load_employees()
        self.orb = cv2.ORB_create(nfeatures=1000)
        def _find_haar():
            cands = ["/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades", "/usr/share/opencv/data/haarcascades", "/usr/local/share/opencv4/haarcascades"]
            for base in cands:
                p = os.path.join(base, "haarcascade_frontalface_default.xml")
                if os.path.exists(p):
                    return base
            return None
        HAAR = _find_haar()
        if not HAAR:
            raise RuntimeError("Brak plików Haar cascades. Zainstaluj: sudo apt install opencv-data")
        def _find_haar():
            cands = ["/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades", "/usr/share/opencv/data/haarcascades", "/usr/local/share/opencv4/haarcascades"]
            for base in cands:
                if base and os.path.exists(os.path.join(base, "haarcascade_frontalface_default.xml")):
                    return base
            return None
        HAAR = _find_haar()
        if not HAAR:
            raise RuntimeError("Nie znaleziono plików Haar cascades (spróbuj: sudo apt install opencv-data).")
        
        HAAR = cv2.data.haarcascades if getattr(cv2.data, "haarcascades", "") else "/usr/share/opencv4/haarcascades/"; self.cascade = cv2.CascadeClassifier(os.path.join(HAAR, "haarcascade_frontalface_default.xml")))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.index = {}  # emp_id → list of descriptors (list[np.ndarray])
        self._load_index()

    def _load_employees(self):
        with open(self.employees_json, "r", encoding="utf-8") as f:
            self.employees = json.load(f)
        # mapy pomocnicze
        self.emp_by_pin = {e["pin"]: e for e in self.employees.get("employees", [])}
        self.emp_by_id = {e.get("id") or e.get("name"): e for e in self.employees.get("employees", [])}

    def save_employees(self):
        with open(self.employees_json, "w", encoding="utf-8") as f:
            json.dump(self.employees, f, ensure_ascii=False, indent=2)
        self._load_employees()

    def add_or_update_employee(self, emp_id: str, name: str, pin: str):
        # wstaw/aktualizuj
        found = False
        for e in self.employees["employees"]:
            if e.get("id") == emp_id:
                e["name"] = name
                e["pin"] = pin
                found = True
                break
        if not found:
            self.employees["employees"].append({"id": emp_id, "name": name, "pin": pin})
        self.save_employees()
        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    def employee_has_images(self, emp_id: str) -> bool:
        folder = os.path.join(self.faces_dir, emp_id)
        return len(glob.glob(os.path.join(folder, "*.jpg"))) > 0

    def _load_index(self):
        self.index.clear()
        for e in self.employees.get("employees", []):
            emp_id = e.get("id") or e.get("name")
            npz_path = os.path.join(self.index_dir, f"{emp_id}.npz")
            if os.path.exists(npz_path):
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    desc_list = list(npz["descriptors"]) if "descriptors" in npz else []
                    self.index[emp_id] = desc_list
                except Exception:
                    self.index[emp_id] = []
            else:
                self.index[emp_id] = []

    def _save_index_for(self, emp_id: str, descriptors_list):
        np.savez_compressed(os.path.join(self.index_dir, f"{emp_id}.npz"), descriptors=np.array(descriptors_list, dtype=object))

    def train_reindex(self, progress_callback=None):
        """Przebuduj indeks ORB dla wszystkich pracowników ("Trening AI")."""
        emps = self.employees.get("employees", [])
        n = len(emps)
        for i, e in enumerate(emps):
            emp_id = e.get("id") or e.get("name")
            folder = os.path.join(self.faces_dir, emp_id)
            desc_list = []
            for imgp in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
                img = cv2.imread(imgp)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # wykryj twarz – jeśli jest, wytnij; jeśli nie, bierz całość
                faces = self.cascade.detectMultiScale(gray, 1.2, 5)
                roi = None
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                    roi = gray[y:y+h, x:x+w]
                else:
                    roi = gray
                roi = cv2.resize(roi, (240, 240), interpolation=cv2.INTER_LINEAR)
                kpts, desc = self.orb.detectAndCompute(roi, None)
                if desc is not None and len(desc) > 0:
                    desc_list.append(desc)
            self.index[emp_id] = desc_list
            self._save_index_for(emp_id, desc_list)
            if progress_callback:
                progress_callback(i + 1, n)

    def replace_oldest_with(self, emp_id: str, img_bgr: np.ndarray):
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if existing:
            oldest = existing[0]
            cv2.imwrite(oldest, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            # jeśli brak – zapisz nowy z timestampem
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            outp = os.path.join(folder, f"{ts}.jpg")
            cv2.imwrite(outp, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def add_three_shots(self, emp_id: str, imgs_bgr_list):
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        for img in imgs_bgr_list:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            outp = os.path.join(folder, f"{ts}.jpg")
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def recognize_face(self, img_bgr: np.ndarray):
        """
        Zwraca: (emp_id or None, display_name, confidence_percent, face_bbox or None)
        Metoda: Haar (detekcja), ORB deskryptory i BFMatcher do wzorców w indeksie.
        Confidence bazuje na liczbie dopasowań do najlepszego pracownika vs. skala.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return None, None, 0.0, None
        # bierz największą twarz
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (240, 240), interpolation=cv2.INTER_LINEAR)
        kpts, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(desc) == 0:
            return None, None, 0.0, (x, y, w, h)

        best_emp = None
        best_score = 0
        second_best = 0

        for emp_id, desc_list in self.index.items():
            emp_score = 0
            for dset in desc_list:
                matches = self.matcher.match(desc, dset)
                if not matches:
                    continue
                # sortuj i policz „dobre” dopasowania (np. odległość < 64)
                good = [m for m in matches if m.distance < 64]
                emp_score += len(good)
            if emp_score > best_score:
                second_best = best_score
                best_score = emp_score
                best_emp = emp_id
            elif emp_score > second_best:
                second_best = emp_score

        # heurystyka confidence: skala do 100 i kontrast do #2
        if best_score == 0:
            conf = 0.0
        else:
            # im większa przewaga nad drugim, tym lepiej
            margin = best_score - second_best
            conf = min(100.0, (best_score * 2.0 + margin) * 2.0)  # skalowanie empiryczne

        display_name = None
        if best_emp and conf > 0:
            e = self.emp_by_id.get(best_emp)
            display_name = e.get("name") if e else best_emp
        return best_emp if conf > 0 else None, display_name, conf, (x, y, w, h)


################################
# GUI + logika stanów aplikacji #
################################

class KeypadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Wprowadź PIN"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: rgba(0,0,0,210); color: white;")
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(title)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 28px; font-weight: 600;")
        layout.addWidget(self.label)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
        self.edit.setFixedHeight(60)
        self.edit.setStyleSheet("font-size: 32px; padding: 8px; border-radius: 12px; background: #222;")
        layout.addWidget(self.edit)
        grid = QtWidgets.QGridLayout()
        btn_style = "font-size:26px; padding:16px; border-radius:16px; background:#333;"
        buttons = [
            ("1", 0, 0), ("2", 0, 1), ("3", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("7", 2, 0), ("8", 2, 1), ("9", 2, 2),
            ("←", 3, 0), ("0", 3, 1), ("OK", 3, 2),
        ]
        for text, r, c in buttons:
            b = QtWidgets.QPushButton(text)
            b.setStyleSheet(btn_style)
            b.clicked.connect(lambda checked=False, t=text: self.on_btn(t))
            grid.addWidget(b, r, c)
        layout.addLayout(grid)
        self.resize(460, 640)

    def on_btn(self, t):
        if t == "OK":
            self.accept()
        elif t == "←":
            txt = self.edit.text()
            self.edit.setText(txt[:-1])
        else:
            self.edit.setText(self.edit.text() + t)

    def value(self):
        return self.edit.text()


class CameraWorker(QtCore.QObject):
    frameReady = Signal(object)

    def __init__(self, width, height, fps):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.picam = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.grab)

    def start(self):
        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(main={"size": (self.width, self.height), "format": "RGB888"})
        self.picam.configure(config)
        self.picam.start()
        self.timer.start(int(1000 / max(1, self.fps)))

    @Slot()
    def grab(self):
        if self.picam is None:
            return
        try:
            frame = self.picam.capture_array("main")  # HxWx3 RGB
            self.frameReady.emit(frame)
        except Exception:
            pass

    def stop(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.picam:
                self.picam.stop()
        except Exception:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Alkotester – Raspberry Pi")
        if CONFIG["fullscreen"]:
            self.showFullScreen()
        self.resize(CONFIG["screen_width"], CONFIG["screen_height"])

        # GPIO – przekaźnik
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(CONFIG["gate_gpio"], GPIO.OUT, initial=GPIO.LOW)

        # MCP3008 + MQ-3
        self.adc = MCP3008(CONFIG["spi_bus"], CONFIG["spi_device"]) 
        self.mq3 = MQ3Sensor(self.adc, CONFIG["mq3_channel"], CONFIG["baseline_samples"], CONFIG["promille_scale"]) 

        # Face DB
        self.facedb = FaceDB(CONFIG["faces_dir"], CONFIG["index_dir"], CONFIG["employees_json"])

        # UI
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.vbox = QtWidgets.QVBoxLayout(self.central)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)

        # Podgląd w QLabel (będzie pełnoekranowy)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background: black;")
        self.vbox.addWidget(self.view, 1)

        # Overlay z komunikatami i przyciskami
        self.overlay = QtWidgets.QFrame()
        self.overlay.setStyleSheet("background: rgba(0, 0, 0, 110);")
        self.ov_layout = QtWidgets.QVBoxLayout(self.overlay)
        self.ov_layout.setContentsMargins(16, 12, 16, 12)

        self.lbl_top = QtWidgets.QLabel("")
        self.lbl_top.setStyleSheet("color: white; font-size: 28px; font-weight: 600;")
        self.ov_layout.addWidget(self.lbl_top)

        self.lbl_center = QtWidgets.QLabel("")
        self.lbl_center.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_center.setStyleSheet("color: white; font-size: 36px; font-weight: 700;")
        self.ov_layout.addWidget(self.lbl_center, 1)

        self.btn_row = QtWidgets.QHBoxLayout()
        self.btn_primary = QtWidgets.QPushButton("Ponów pomiar")
        self.btn_primary.setStyleSheet("font-size: 24px; padding: 12px 18px; border-radius: 16px; background: #2e7d32; color: white;")
        self.btn_secondary = QtWidgets.QPushButton("Wprowadź PIN")
        self.btn_secondary.setStyleSheet("font-size: 24px; padding: 12px 18px; border-radius: 16px; background: #1565c0; color: white;")
        self.btn_row.addWidget(self.btn_primary)
        self.btn_row.addWidget(self.btn_secondary)
        self.ov_layout.addLayout(self.btn_row)

        self.vbox.addWidget(self.overlay, 0)

        # Stany
        self.state = "idle"  # idle -> detect -> recognized/pin -> calibrate -> measure -> decide
        self.current_emp_id = None
        self.current_emp_name = None
        self.fallback_pin_flag = False
        self.last_face_bbox = None

        # Kamera
        self.cam_worker = CameraWorker(CONFIG["camera_main_size"][0], CONFIG["camera_main_size"][1], CONFIG["camera_fps"])
        self.cam_thread = QtCore.QThread()
        self.cam_worker.moveToThread(self.cam_thread)
        self.cam_worker.frameReady.connect(self.on_frame)
        self.cam_thread.started.connect(self.cam_worker.start)
        self.cam_thread.start()

        # Timery
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.on_tick)
        self.ui_timer.start(250)

        # Zdarzenia UI
        self.btn_primary.clicked.connect(self.on_btn_primary)
        self.btn_secondary.clicked.connect(self.on_btn_secondary)

        # Kalibracja MQ3 na starcie
        self.set_message("Proszę czekać…", "Kalibracja czujnika MQ-3 w toku")
        QtCore.QTimer.singleShot(10, self._calibrate_mq3_start)

        # Tryb początkowy
        self.next_idle()

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.cam_worker.stop()
            self.cam_thread.quit()
            self.cam_thread.wait(1000)
        except Exception:
            pass
        try:
            self.adc.close()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        return super().closeEvent(e)

    ########################
    # Pomocnicze / UI tekst #
    ########################
    def set_message(self, top, center=None):
        self.lbl_top.setText(top)
        self.lbl_center.setText(center or "")

    def show_buttons(self, primary=False, secondary=False, p_text=None, s_text=None):
        self.btn_primary.setVisible(primary)
        self.btn_secondary.setVisible(secondary)
        if p_text:
            self.btn_primary.setText(p_text)
        if s_text:
            self.btn_secondary.setText(s_text)

    ########################
    # Cykl życia stanów     #
    ########################
    def next_idle(self):
        self.state = "idle"
        self.current_emp_id = None
        self.current_emp_name = None
        self.fallback_pin_flag = False
        self.set_message(now_str(), "Podejdź bliżej")
        self.show_buttons(primary=False, secondary=True, s_text="Wprowadź PIN")

    def goto_detect(self):
        self.state = "detect"
        self.set_message(now_str(), "Szukam twarzy…")
        self.show_buttons(primary=False, secondary=True, s_text="Wprowadź PIN")

    def goto_calibrate(self):
        self.state = "calibrate"
        self.set_message("Proszę czekać…", "Kalibracja")
        self.show_buttons(primary=False, secondary=False)
        # krótka kalibracja – sprawdź czy twarz jest wystarczająco blisko (wielkość bbox)
        QtCore.QTimer.singleShot(1200, self.after_calibration)

    def goto_measure(self):
        self.state = "measure"
        self.set_message("A dmuchnij no…", "3 s pomiaru")
        self.show_buttons(primary=False, secondary=False)
        # uruchom rzeczywisty pomiar
        threading.Thread(target=self._do_measurement_thread, daemon=True).start()

    def goto_decide(self, promille):
        prom_s = f"Pomiar: {promille:.3f} [‰]"
        self.set_message(now_str(), prom_s)
        if promille <= CONFIG["threshold_pass"]:
            # otwórz przejście
            self.set_message(prom_s, "Przejście otwarte")
            self.trigger_gate_and_log(pass_ok=True, promille=promille)
            QtCore.QTimer.singleShot(2500, self.next_idle)
        elif promille < CONFIG["threshold_deny"]:
            # ponów pomiar
            self.show_buttons(primary=True, secondary=True, p_text="Ponów pomiar", s_text="Odmowa")
            self.state = "retry"
            self._last_promille = promille
        else:
            # odmowa
            self.set_message(prom_s, "Odmowa")
            self.trigger_gate_and_log(pass_ok=False, promille=promille)
            QtCore.QTimer.singleShot(3000, self.next_idle)

    def after_calibration(self):
        # warunek: jeśli jest bbox twarzy i jest sensownie duży – sukces
        ok = False
        if self.last_face_bbox is not None:
            (x, y, w, h) = self.last_face_bbox
            ok = max(w, h) >= CONFIG["face_min_size"]
        if ok:
            self.goto_measure()
        else:
            # kalibracja nieudana → PIN fallback
            self.fallback_pin_flag = True
            self.set_message("Kalibracja nie udana", "Przejście do pomiaru na PIN")
            # przekazujemy „flagę” do logów/serwera w czasie decyzji
            QtCore.QTimer.singleShot(1200, self.ask_pin)

    def ask_pin(self):
        dlg = KeypadDialog(self, title="Wprowadź PIN")
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            pin = dlg.value()
            emp = self.facedb.emp_by_pin.get(pin)
            if emp:
                self.current_emp_id = emp.get("id") or emp.get("name")
                self.current_emp_name = emp.get("name")
                self.set_message(f"Cześć {self.current_emp_name}", "Gotowy do pomiaru")
                QtCore.QTimer.singleShot(800, self.goto_measure)
            else:
                # nowy pracownik – utwórz konto
                name = f"user_{pin}"
                emp_id = name
                self.facedb.add_or_update_employee(emp_id, name, pin)
                self.current_emp_id = emp_id
                self.current_emp_name = name
                # Wariant A – zrób 3 zdjęcia co 1s
                self.variant_A_collect_new()
        else:
            self.next_idle()

    def variant_A_collect_new(self):
        self.set_message("Proszę czekać…", "Zapis 3 zdjęć (1/s)")
        imgs = []
        def snap_once(i):
            if self.last_frame_bgr is None:
                return
            img = self.last_frame_bgr.copy()
            imgs.append(img)
            if i < 3:
                QtCore.QTimer.singleShot(1000, lambda: snap_once(i + 1))
            else:
                self.facedb.add_three_shots(self.current_emp_id, imgs)
                self.training_progress("Trening AI")
        snap_once(1)

    def variant_B_replace_old(self):
        # zastąpienie najstarszego zdjęcia bieżącą klatką
        if self.last_frame_bgr is not None and self.current_emp_id:
            self.facedb.replace_oldest_with(self.current_emp_id, self.last_frame_bgr.copy())
            self.training_progress("Trening AI")

    def training_progress(self, title):
        self.set_message("Proszę czekać…", title)
        self.show_buttons(primary=False, secondary=False)
        def cb(done, total):
            self.set_message(f"{title}", f"{done}/{total}")
        threading.Thread(target=self.facedb.train_reindex, kwargs={"progress_callback": cb}, daemon=True).start()
        # po krótkiej pauzie – test po aktualizacji
        QtCore.QTimer.singleShot(2200, self.after_training)

    def after_training(self):
        self.set_message("Test po aktualizacji", "Gotowy")
        QtCore.QTimer.singleShot(800, self.goto_detect)

    def trigger_gate_and_log(self, pass_ok: bool, promille: float):
        emp = self.current_emp_name or "<nieznany>"
        ts = datetime.now()
        # sygnał GPIO tylko przy pass_ok
        if pass_ok:
            GPIO.output(CONFIG["gate_gpio"], GPIO.HIGH)
            def pulse():
                time.sleep(CONFIG["gate_pulse_sec"])
                GPIO.output(CONFIG["gate_gpio"], GPIO.LOW)
            threading.Thread(target=pulse, daemon=True).start()
            # „Upload otwarcia bramki na serwer …” – tutaj placeholder
            log_csv(os.path.join(CONFIG["logs_dir"], "events.csv"),
                    ["datetime", "event", "employee"],
                    [ts.isoformat(), "gate_open", emp])
        else:
            # Raport do pracodawcy
            log_csv(os.path.join(CONFIG["logs_dir"], "events.csv"),
                    ["datetime", "event", "employee"],
                    [ts.isoformat(), "deny_access", emp])
        # log pomiaru
        log_csv(os.path.join(CONFIG["logs_dir"], "measurements.csv"),
                ["datetime", "employee", "promille", "fallback_pin"],
                [ts.isoformat(), emp, f"{promille:.3f}", int(self.fallback_pin_flag)])

    ########################
    # Obsługa przycisków    #
    ########################
    def on_btn_primary(self):
        if self.state == "retry":
            self.goto_measure()

    def on_btn_secondary(self):
        if self.state in ("idle", "detect", "retry"):
            self.ask_pin()

    ########################
    # Kamera / ramki         #
    ########################
    def on_frame(self, frame_rgb: np.ndarray):
        # zapis ostatniej klatki w BGR do obróbki (OpenCV)
        self.last_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # prosty overlay w zależności od stanu (np. bbox twarzy)
        disp = self.last_frame_bgr.copy()

        if self.state in ("idle", "detect", "calibrate"):
            emp_id, emp_name, conf, bbox = self.facedb.recognize_face(disp)
            self.last_face_bbox = bbox
            if bbox is not None:
                (x, y, w, h) = bbox
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if self.state == "detect":
                if emp_name and conf >= CONFIG["recognition_conf_ok"]:
                    self.current_emp_id = emp_id
                    self.current_emp_name = emp_name
                    self.set_message(f"Cześć {emp_name}", f"pewność: {conf:.0f}%")
                    # Wariant B: pracownik istnieje – zrób dodatkowe zdjęcie i trenuj
                    self.variant_B_replace_old()
                    QtCore.QTimer.singleShot(1000, self.goto_calibrate)
                else:
                    if conf <= CONFIG["recognition_conf_low"]:
                        self.set_message(now_str(), "Nie rozpoznaję …")
                    else:
                        self.set_message(now_str(), f"pewność: {conf:.0f}%")
        # rotacja do portretu – zależnie od konfiguracji
        if CONFIG["rotate_180"]:
            disp = cv2.rotate(disp, cv2.ROTATE_180)

        # dopasuj do rozmiaru okna i pokaż
        h, w, _ = disp.shape
        target_w, target_h = CONFIG["screen_width"], CONFIG["screen_height"]
        # obróć do portretu (jeśli strumień jest landscape 1280x720)
        if w > h:
            disp = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)
        disp_resized = cv2.resize(disp, (target_w, target_h), interpolation=cv2.INTER_AREA)
        qimg = QtGui.QImage(disp_resized.data, target_w, target_h, 3 * target_w, QtGui.QImage.Format_BGR888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg))

        # automatyczne przejście z idle do detect kiedy pojawi się twarz
        if self.state == "idle":
            # szybka sonda detekcji – jeśli bbox jest, przejdź do detect
            gray = cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.facedb.cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                self.goto_detect()

    def on_tick(self):
        # odświeżaj zegar w idle/detect
        if self.state in ("idle", "detect"):
            self.lbl_top.setText(now_str())

    ########################
    # Pomiar MQ-3 (wątki)    #
    ########################
    def _calibrate_mq3_start(self):
        def work():
            self.mq3.calibrate_baseline()
        t = threading.Thread(target=work, daemon=True)
        t.start()
        QtCore.QTimer.singleShot(1200, self.set_message, args=(now_str(), "Podejdź bliżej"))

    def _do_measurement_thread(self):
        # 3s pomiaru
        t_end = time.time() + CONFIG["measure_seconds"]
        while time.time() < t_end:
            left = t_end - time.time()
            self.set_message("A dmuchnij no…", f"{left:0.1f} s")
            time.sleep(0.1)
        promille, raw = self.mq3.read_promille(duration_sec=0.5)
        # decyzja
        QtCore.QTimer.singleShot(0, lambda p=promille: self.goto_decide(p))


############################
# MAIN                     #
############################

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # tło półprzezroczyste overlay na dole – układ już dodany (overlay na końcu vbox)
    if CONFIG["fullscreen"]:
        win.showFullScreen()
    else:
        win.show()

    # Obsługa SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
