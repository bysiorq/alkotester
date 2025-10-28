#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alkotester – Raspberry Pi 4 kiosk (DSI 720x1280)
------------------------------------------------

- Fullscreen kiosk na panelu dotykowym DSI 720x1280 (pion).
- Podgląd kamery Picamera2:
  - łapiemy 1280x720 (landscape),
  - obracamy softowo (domyślnie 90° w prawo = "cw"),
  - cropujemy środek i skalujemy tak żeby WYPEŁNIĆ obszar nad paskiem przycisków
    bez czarnych pasów i bez rozciągania ludzi.
- Kolory poprawne: pracujemy wewnętrznie w BGR i dopiero przed wyświetleniem
  zmieniamy na RGB dla Qt.
- Ramka twarzy + procent pewności rozpoznania w prawym dolnym rogu tej ramki.
- Stany: idle → detect → calibrate → measure ("A dmuchnij no…") → decide / retry.
- PIN fallback.
- MCP3008 + MQ-3 (SPI0) → pseudo-promile.
- Logi CSV + impuls GPIO18 (otwarcie przejścia).
- Pracownik testowy: id="1", name="Kamil Karolak", pin="0000".
- Automatycznie ustawiamy DISPLAY i resztę zmiennych Qt, żeby działało nawet
  przy uruchamianiu przez SSH.

Wymagane pakiety:
    sudo apt update && sudo apt install -y \
        python3-picamera2 python3-opencv python3-pyqt5 python3-numpy \
        python3-spidev python3-rpi.gpio
"""

import os
import sys
import cv2
import json
import time
import glob
import signal
import threading
import numpy as np
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
Signal = QtCore.pyqtSignal
Slot   = QtCore.pyqtSlot

from picamera2 import Picamera2
import spidev
import RPi.GPIO as GPIO


# =========================
# KONFIG
# =========================
CONFIG = {
    # fizyczna rozdziałka panelu DSI (pionowo)
    "screen_width": 720,
    "screen_height": 1280,

    # pasek z komunikatem + przyciskami na dole
    "overlay_height_px": 220,

    # tryb kiosk
    # UWAGA: kiosk_window_flags=True dawało Ci okno małe w rogu -> wyłączamy.
    "fullscreen": True,
    "kiosk_window_flags": False,
    "hide_cursor": False,

    # Kamera:
    # capture w landscape (1280x720),
    # rotate_dir potem obraca numpy:
    #   "cw"   = 90° w prawo (to chcemy mieć pion)
    #   "ccw"  = 90° w lewo
    #   "180"  = do góry nogami
    #   "none" = bez obrotu (poziomo)
    "camera_main_size": (1280, 720),  # (W,H) z sensora
    "camera_fps": 30,
    "rotate_dir": "cw",

    # rozpoznawanie twarzy
    "face_detect_interval_ms": 1000,
    "face_min_size": 120,
    "recognition_conf_ok": 55.0,   # %
    "recognition_conf_low": 20.0,  # %

    # MQ-3 / MCP3008
    "spi_bus": 0,
    "spi_device": 0,          # CE0
    "mq3_channel": 0,
    "baseline_samples": 150,
    "promille_scale": 220.0,
    "measure_seconds": 3.0,

    # progi decyzji [‰]
    "threshold_pass": 0.00,
    "threshold_retry": 0.20,
    "threshold_deny": 0.50,

    # przekaźnik bramki
    "gate_gpio": 18,
    "gate_pulse_sec": 5.0,

    # pliki danych
    "data_dir": "data",
    "faces_dir": "data/faces",
    "index_dir": "data/index",
    "employees_json": "data/employees.json",
    "logs_dir": "logs",

    # pierwszy pracownik żeby baza nie była pusta
    "bootstrap_employee": {
        "id": "1",
        "name": "Kamil Karolak",
        "pin": "0000",
    },

    "debug": False,
}


############################
# UTIL: katalogi / logi / czas
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
# MCP3008 / MQ-3
############################
class MCP3008:
    def __init__(self, bus=0, device=0, max_speed_hz=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = 0

    def read_channel(self, ch: int) -> int:
        if ch < 0 or ch > 7:
            raise ValueError("MCP3008 channel 0..7 only")
        r = self.spi.xfer2([1, (8 | ch) << 4, 0])
        return ((r[1] & 3) << 8) | r[2]  # 0..1023

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
        samples = [self.adc.read_channel(self.channel) for _ in range(self.baseline_samples)]
        self.baseline = float(np.median(samples))
        return self.baseline

    def read_raw(self):
        return self.adc.read_channel(self.channel)

    def promille_from_samples(self, samples):
        v = float(np.mean(samples)) if samples else float(self.read_raw())
        if self.baseline is None:
            self.baseline = v
        delta = max(0.0, v - self.baseline)
        return delta / max(1e-6, self.promille_scale)


############################
# FaceDB (Haar + ORB)
############################
class FaceDB:
    """
    employees.json:
      {"employees":[{"id":"1","name":"Kamil Karolak","pin":"0000"}, ...]}

    faces/<id>/*.jpg – zdjęcia pracownika
    index/<id>.npz   – deskryptory ORB dla danego pracownika
    """
    def __init__(self, faces_dir, index_dir, employees_json):
        self.faces_dir = faces_dir
        self.index_dir = index_dir
        self.employees_json = employees_json

        self._load_employees()
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.cascade = cv2.CascadeClassifier(self._find_haar())
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.index = {}
        self._load_index()

    def _find_haar(self):
        candidates = []
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidates.append(cv2.data.haarcascades)
        candidates += ["/usr/share/opencv4/haarcascades/", "/usr/share/opencv/haarcascades/"]
        fname = "haarcascade_frontalface_default.xml"
        for base in candidates:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        return fname

    def _load_employees(self):
        with open(self.employees_json, "r", encoding="utf-8") as f:
            self.employees = json.load(f)
        self.emp_by_pin = {e["pin"]: e for e in self.employees.get("employees", []) if "pin" in e}
        self.emp_by_id  = {e.get("id") or e.get("name"): e for e in self.employees.get("employees", [])}

    def save_employees(self):
        with open(self.employees_json, "w", encoding="utf-8") as f:
            json.dump(self.employees, f, ensure_ascii=False, indent=2)
        self._load_employees()

    def ensure_employee_exists(self, emp_id: str, name: str, pin: str):
        if not any(e.get("id") == emp_id for e in self.employees["employees"]):
            self.employees["employees"].append({"id": emp_id, "name": name, "pin": pin})
            self.save_employees()
        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    def add_or_update_employee(self, emp_id: str, name: str, pin: str):
        for e in self.employees["employees"]:
            if e.get("id") == emp_id:
                e["name"] = name
                e["pin"]  = pin
                self.save_employees()
                break
        else:
            self.employees["employees"].append({"id": emp_id, "name": name, "pin": pin})
            self.save_employees()
        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    def _load_index(self):
        self.index = {}
        for e in self.employees.get("employees", []):
            emp_id = e.get("id") or e.get("name")
            npz_path = os.path.join(self.index_dir, f"{emp_id}.npz")
            if os.path.exists(npz_path):
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    self.index[emp_id] = list(npz["descriptors"]) if "descriptors" in npz else []
                except Exception:
                    self.index[emp_id] = []
            else:
                self.index[emp_id] = []

    def _save_index_for(self, emp_id: str, descriptors_list):
        os.makedirs(self.index_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.index_dir, f"{emp_id}.npz"),
            descriptors=np.array(descriptors_list, dtype=object)
        )

    def train_reindex(self, progress_callback=None):
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
                faces = self.cascade.detectMultiScale(gray, 1.2, 5)

                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                    roi = gray[y:y+h, x:x+w]
                else:
                    roi = gray

                roi = cv2.resize(roi, (240, 240), interpolation=cv2.INTER_LINEAR)
                _, desc = self.orb.detectAndCompute(roi, None)
                if desc is not None and len(desc) > 0:
                    desc_list.append(desc)

            self.index[emp_id] = desc_list
            self._save_index_for(emp_id, desc_list)
            if progress_callback:
                progress_callback(i+1, n)

    def replace_oldest_with(self, emp_id: str, img_bgr):
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if existing:
            outp = existing[0]
        else:
            outp = os.path.join(folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
        cv2.imwrite(outp, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def add_three_shots(self, emp_id: str, imgs_bgr_list):
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        for img in imgs_bgr_list:
            outp = os.path.join(folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def recognize_face(self, img_bgr):
        """
        img_bgr: klatka już po obrocie, tak jak użytkownik ją widzi.
        Return:
          (emp_id or None,
           display_name or None,
           confidence%, 
           bbox (x,y,w,h) or None)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
            return None, None, 0.0, None

        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

        roi = cv2.resize(gray[y:y+h, x:x+w], (240, 240), interpolation=cv2.INTER_LINEAR)
        _, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(desc) == 0:
            return None, None, 0.0, (x, y, w, h)

        best_emp, best_score, second_best = None, 0, 0
        for emp_id, d_list in self.index.items():
            emp_score = 0
            for d in d_list:
                matches = self.matcher.match(desc, d)
                if matches:
                    good = [m for m in matches if m.distance < 64]
                    emp_score += len(good)
            if emp_score > best_score:
                second_best, best_score, best_emp = best_score, emp_score, emp_id
            elif emp_score > second_best:
                second_best = emp_score

        if best_score == 0:
            conf = 0.0
        else:
            margin = best_score - second_best
            conf = min(100.0, (best_score * 2.0 + margin) * 2.0)

        display_name = self.emp_by_id.get(best_emp, {}).get("name") if best_emp else None
        return (best_emp if conf > 0 else None), display_name, conf, (x, y, w, h)


############################
# PIN keypad
############################
class KeypadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Wprowadź PIN"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: rgba(0,0,0,210); color: white;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16,16,16,16)

        lbl = QtWidgets.QLabel(title)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("font-size:28px; font-weight:600; color:white;")
        layout.addWidget(lbl)

        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
        self.edit.setFixedHeight(60)
        self.edit.setStyleSheet(
            "font-size:32px; padding:8px; border-radius:12px; background:#222; color:white;"
        )
        layout.addWidget(self.edit)

        grid = QtWidgets.QGridLayout()
        btnstyle = (
            "font-size:26px; padding:16px; border-radius:16px; "
            "background:#333; color:white;"
        )
        keys = [
            ("1",0,0),("2",0,1),("3",0,2),
            ("4",1,0),("5",1,1),("6",1,2),
            ("7",2,0),("8",2,1),("9",2,2),
            ("←",3,0),("0",3,1),("OK",3,2),
        ]
        for t,r,c in keys:
            b = QtWidgets.QPushButton(t)
            b.setStyleSheet(btnstyle)
            b.clicked.connect(lambda _,x=t:self.on_btn(x))
            grid.addWidget(b,r,c)
        layout.addLayout(grid)

        self.resize(460,640)

    def on_btn(self, t):
        if t == "OK":
            self.accept()
        elif t == "←":
            self.edit.setText(self.edit.text()[:-1])
        else:
            self.edit.setText(self.edit.text() + t)

    def value(self):
        return self.edit.text()


############################
# Camera manager
############################
class CameraManager:
    """
    Picamera2.capture_array("main") daje RGB888.
    U Ciebie obraz w przeszłości wyglądał jakby kanały były poprzestawiane
    (mocno niebieski). Żeby to naprawić:
    - przyjmujemy, że to co dostajemy tak naprawdę pasuje do BGR,
      więc traktujemy bufor jako BGR i dopiero tuż przed wyświetleniem
      robimy cvtColor(BGR->RGB) dla Qt.
    """

    def __init__(self, width, height, rotate_dir):
        self.rotate_dir = rotate_dir
        self.picam = Picamera2()
        cfg = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam.configure(cfg)
        self.picam.start()

    def get_frame_bgr(self):
        frame = self.picam.capture_array("main")  # numpy (H,W,3)
        # obrót numpy.rot90:
        if self.rotate_dir == "cw":       # 90° w prawo
            frame = np.rot90(frame, 3)
        elif self.rotate_dir == "ccw":    # 90° w lewo
            frame = np.rot90(frame, 1)
        elif self.rotate_dir == "180":
            frame = np.rot90(frame, 2)
        # teraz frame traktujemy jako BGR
        return frame

    def stop(self):
        try:
            self.picam.stop()
        except Exception:
            pass


############################
# MainWindow
############################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        # GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(CONFIG["gate_gpio"], GPIO.OUT, initial=GPIO.LOW)

        # MQ-3
        self.adc = MCP3008(CONFIG["spi_bus"], CONFIG["spi_device"])
        self.mq3 = MQ3Sensor(
            self.adc,
            CONFIG["mq3_channel"],
            CONFIG["baseline_samples"],
            CONFIG["promille_scale"],
        )

        # Face DB
        self.facedb = FaceDB(
            CONFIG["faces_dir"],
            CONFIG["index_dir"],
            CONFIG["employees_json"]
        )
        boot = CONFIG["bootstrap_employee"]
        self.facedb.ensure_employee_exists(boot["id"], boot["name"], boot["pin"])

        # Okno kiosk
        self.setWindowTitle("Alkotester – Raspberry Pi")
        # kiosk_window_flags=False domyślnie, bo True rozwalało geometrię na Twoim RPi
        if CONFIG["kiosk_window_flags"]:
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.X11BypassWindowManagerHint
            )
        if CONFIG["hide_cursor"]:
            self.setCursor(QtCore.Qt.BlankCursor)

        # GŁÓWNY LAYOUT:
        #   [ widok kamery (dynamiczny) ]
        #   [ overlay (220px) ]
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(0)

        # podgląd kamery
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background:black;")
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        vbox.addWidget(self.view, 1)

        # overlay (pasek stanu i przycisków)
        self.overlay = QtWidgets.QFrame()
        self.overlay.setFixedHeight(CONFIG["overlay_height_px"])
        self.overlay.setStyleSheet("background: rgba(0,0,0,110); color:white;")

        ov = QtWidgets.QVBoxLayout(self.overlay)
        ov.setContentsMargins(16,12,16,12)
        ov.setSpacing(8)

        self.lbl_top = QtWidgets.QLabel("")
        self.lbl_top.setStyleSheet("color:white; font-size:28px; font-weight:600;")
        ov.addWidget(self.lbl_top)

        self.lbl_center = QtWidgets.QLabel("")
        self.lbl_center.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_center.setStyleSheet("color:white; font-size:36px; font-weight:700;")
        ov.addWidget(self.lbl_center, 1)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)

        self.btn_primary = QtWidgets.QPushButton("Ponów pomiar")
        self.btn_primary.setStyleSheet(
            "font-size:24px; padding:12px 18px; border-radius:16px; background:#2e7d32; color:white;"
        )
        self.btn_secondary = QtWidgets.QPushButton("Wprowadź PIN")
        self.btn_secondary.setStyleSheet(
            "font-size:24px; padding:12px 18px; border-radius:16px; background:#1565c0; color:white;"
        )

        row.addWidget(self.btn_primary)
        row.addWidget(self.btn_secondary)
        ov.addLayout(row)

        vbox.addWidget(self.overlay, 0)

        # Stan runtime
        self.state = "idle"
        self.current_emp_id = None
        self.current_emp_name = None
        self.fallback_pin_flag = False
        self.last_face_bbox = None
        self.last_confidence = 0.0

        self.frame_last_bgr = None
        self.measure_timer = None
        self.measure_deadline = 0
        self.measure_samples = []

        # Kamera
        self.cam = CameraManager(
            CONFIG["camera_main_size"][0],
            CONFIG["camera_main_size"][1],
            CONFIG["rotate_dir"],
        )

        # Timery
        self.cam_timer  = QtCore.QTimer()
        self.cam_timer.timeout.connect(self.on_camera_tick)
        self.cam_timer.start(int(1000/max(1,CONFIG["camera_fps"])))

        self.ui_timer   = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.on_ui_tick)
        self.ui_timer.start(250)

        self.face_timer = QtCore.QTimer()
        self.face_timer.timeout.connect(self.face_recognition_tick)
        self.face_timer.start(CONFIG["face_detect_interval_ms"])

        # Przyciski
        self.btn_primary.clicked.connect(self.on_btn_primary)
        self.btn_secondary.clicked.connect(self.on_btn_secondary)

        # Start MQ-3 baseline
        self.set_message("Proszę czekać…", "Kalibracja czujnika MQ-3 w toku")
        self._calibrate_mq3_start()

        # Początkowy stan
        self.next_idle()

    ################################
    # Pomocnicze UI
    ################################
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

    ################################
    # State machine
    ################################
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
        QtCore.QTimer.singleShot(1200, self.after_calibration)

    def after_calibration(self):
        ok = False
        if self.last_face_bbox is not None:
            (_, _, w, h) = self.last_face_bbox
            ok = max(w, h) >= CONFIG["face_min_size"]

        if ok:
            self.goto_measure()
        else:
            self.fallback_pin_flag = True
            self.set_message("Kalibracja nie udana", "Pomiar po PIN")
            QtCore.QTimer.singleShot(800, self.ask_pin)

    def goto_measure(self):
        self.state = "measure"
        self.show_buttons(primary=False, secondary=False)
        self.measure_samples = []
        self.measure_deadline = time.time() + CONFIG["measure_seconds"]

        self.measure_timer = QtCore.QTimer()
        self.measure_timer.timeout.connect(self.measure_tick)
        self.measure_timer.start(100)

        self.set_message("A dmuchnij no…", f"{CONFIG['measure_seconds']:.1f} s")

    def goto_decide(self, promille):
        prom_s = f"Pomiar: {promille:.3f} [‰]"
        if promille <= CONFIG["threshold_pass"]:
            self.set_message(prom_s, "Przejście otwarte")
            self.trigger_gate_and_log(True, promille)
            QtCore.QTimer.singleShot(2500, self.next_idle)
            self.state = "decide"
        elif promille < CONFIG["threshold_deny"]:
            self.set_message(prom_s, "Ponów pomiar?")
            self.show_buttons(primary=True, secondary=True,
                              p_text="Ponów pomiar", s_text="Odmowa")
            self.state = "retry"
        else:
            self.set_message(prom_s, "Odmowa")
            self.trigger_gate_and_log(False, promille)
            QtCore.QTimer.singleShot(3000, self.next_idle)
            self.state = "decide"

    ################################
    # Pomiar MQ-3 tick
    ################################
    def measure_tick(self):
        left = self.measure_deadline - time.time()
        self.measure_samples.append(self.mq3.read_raw())

        if left > 0:
            self.set_message("A dmuchnij no…", f"{left:0.1f} s")
            return

        try:
            self.measure_timer.stop()
        except Exception:
            pass
        self.measure_timer = None

        promille = self.mq3.promille_from_samples(self.measure_samples)
        self.goto_decide(promille)

    ################################
    # PIN fallback
    ################################
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
                # nowy użytkownik → dodaj
                name = f"user_{pin}"
                emp_id = name
                self.facedb.add_or_update_employee(emp_id, name, pin)
                self.current_emp_id = emp_id
                self.current_emp_name = name
                self.variant_A_collect_new()
        else:
            self.next_idle()

    def variant_A_collect_new(self):
        self.set_message("Proszę czekać…", "Zapis 3 zdjęć")
        self.show_buttons(primary=False, secondary=False)

        imgs = []

        def snap(i):
            if self.frame_last_bgr is not None:
                imgs.append(self.frame_last_bgr.copy())
            if i < 3:
                QtCore.QTimer.singleShot(1000, lambda: snap(i+1))
            else:
                self.facedb.add_three_shots(self.current_emp_id, imgs)
                self.training_start()

        snap(1)

    def variant_B_replace_old(self):
        if self.frame_last_bgr is not None and self.current_emp_id:
            self.facedb.replace_oldest_with(
                self.current_emp_id,
                self.frame_last_bgr.copy()
            )
            self.training_start()

    def training_start(self):
        self.set_message("Proszę czekać…", "Trening AI")
        self.show_buttons(primary=False, secondary=False)

        def worker():
            self.facedb.train_reindex()
            QtCore.QMetaObject.invokeMethod(
                self,
                "_training_done",
                QtCore.Qt.QueuedConnection
            )

        threading.Thread(target=worker, daemon=True).start()

    @Slot()
    def _training_done(self):
        self.set_message("Test po aktualizacji", "Gotowy")
        QtCore.QTimer.singleShot(800, self.goto_detect)

    ################################
    # bramka + logi
    ################################
    def trigger_gate_and_log(self, pass_ok: bool, promille: float):
        emp = self.current_emp_name or "<nieznany>"
        ts = datetime.now()

        if pass_ok:
            GPIO.output(CONFIG["gate_gpio"], GPIO.HIGH)

            def pulse():
                time.sleep(CONFIG["gate_pulse_sec"])
                GPIO.output(CONFIG["gate_gpio"], GPIO.LOW)

            threading.Thread(target=pulse, daemon=True).start()

            log_csv(
                os.path.join(CONFIG["logs_dir"], "events.csv"),
                ["datetime","event","employee"],
                [ts.isoformat(),"gate_open",emp]
            )
        else:
            log_csv(
                os.path.join(CONFIG["logs_dir"], "events.csv"),
                ["datetime","event","employee"],
                [ts.isoformat(),"deny_access",emp]
            )

        log_csv(
            os.path.join(CONFIG["logs_dir"], "measurements.csv"),
            ["datetime","employee","promille","fallback_pin"],
            [ts.isoformat(),emp,f"{promille:.3f}",int(self.fallback_pin_flag)]
        )

    ################################
    # obsługa przycisków
    ################################
    def on_btn_primary(self):
        if self.state == "retry":
            self.goto_measure()

    def on_btn_secondary(self):
        if self.state == "retry":
            self.set_message("Odmowa","")
            QtCore.QTimer.singleShot(2000, self.next_idle)
        elif self.state in ("idle","detect"):
            self.ask_pin()

    ################################
    # crop-fill helper
    ################################
    def _crop_and_scale_fill(self, src_rgb, target_w, target_h):
        """
        Wypełnij cały widget bez czarnych pasów:
        1. bierz środek obrazu,
        2. przytnij tak, by aspect == target_w/target_h,
        3. przeskaluj do dokładnie (target_w, target_h).

        Dzięki temu nie ma letterboxa ani pillarboxa.
        Minimalne zniekształcenie = brak, tylko crop.
        """
        if target_w <= 0 or target_h <= 0:
            return None

        sh, sw, _ = src_rgb.shape
        target_aspect = target_w / float(target_h)
        src_aspect = sw / float(sh)

        # jeśli obraz "szerszy" niż chcemy → przytnij szerokość
        if src_aspect > target_aspect:
            # potrzebna szerokość po cropie
            new_sw = int(target_aspect * sh)
            if new_sw > sw:
                new_sw = sw
            x0 = (sw - new_sw) // 2
            cropped = src_rgb[:, x0:x0+new_sw, :]
        else:
            # obraz "węższy/twardszy w pionie" → przytnij wysokość
            new_sh = int(sw / target_aspect)
            if new_sh > sh:
                new_sh = sh
            y0 = (sh - new_sh) // 2
            cropped = src_rgb[y0:y0+new_sh, :, :]

        # przeskaluj dokładnie do target_u
        fitted = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return fitted

    ################################
    # kamera tick (~30 fps)
    ################################
    def on_camera_tick(self):
        frame_bgr = self.cam.get_frame_bgr()
        if frame_bgr is None:
            return

        # zapamiętujemy tę ramkę do face rec / zapisu do bazy
        self.frame_last_bgr = frame_bgr.copy()

        # dorysowujemy bbox + % pewności
        disp_bgr = frame_bgr.copy()

        if self.last_face_bbox is not None:
            (x, y, w, h) = self.last_face_bbox
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

            # kolor ramki w zależności od pewności
            if self.last_confidence >= CONFIG["recognition_conf_ok"]:
                color = (0,255,0)        # zielony
            elif self.last_confidence <= CONFIG["recognition_conf_low"]:
                color = (0,255,255)      # żółty
            else:
                color = (255,255,0)      # cyjan-ish / jasny

            cv2.rectangle(disp_bgr, (x1,y1), (x2,y2), color, 2)
            txt = f"{self.last_confidence:.0f}%"
            cv2.putText(
                disp_bgr,
                txt,
                (x2-10, y2-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # BGR -> RGB przed pokazaniem w Qt
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

        # Dopasuj do aktualnego rozmiaru widżetu KAMERY (bez overlayu),
        # ale BEZ pasów → crop-then-scale.
        target_w = self.view.width()
        target_h = self.view.height()
        fitted = self._crop_and_scale_fill(disp_rgb, target_w, target_h)
        if fitted is None:
            return

        h, w, _ = fitted.shape
        qimg = QtGui.QImage(
            fitted.data,
            w,
            h,
            3*w,
            QtGui.QImage.Format_RGB888
        )
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg))

    ################################
    # face tick (co 1s)
    ################################
    def face_recognition_tick(self):
        if self.frame_last_bgr is None:
            return
        if self.state not in ("idle","detect","calibrate"):
            return

        emp_id, emp_name, conf, bbox = self.facedb.recognize_face(self.frame_last_bgr)
        self.last_face_bbox = bbox
        self.last_confidence = conf or 0.0

        if self.state == "idle" and bbox is not None:
            self.goto_detect()
            return

        if self.state == "detect":
            if emp_name and conf >= CONFIG["recognition_conf_ok"]:
                self.current_emp_id = emp_id
                self.current_emp_name = emp_name
                self.set_message(f"Cześć {emp_name}", f"pewność: {conf:.0f}%")
                self.variant_B_replace_old()
                QtCore.QTimer.singleShot(800, self.goto_calibrate)
            else:
                if conf <= CONFIG["recognition_conf_low"]:
                    self.set_message(now_str(), "Nie rozpoznaję…")
                else:
                    self.set_message(now_str(), f"pewność: {conf:.0f}%")

        # w stanie "calibrate": zostawiamy komunikat "Kalibracja"

    ################################
    # UI tick
    ################################
    def on_ui_tick(self):
        if self.state in ("idle","detect"):
            self.lbl_top.setText(now_str())

    ################################
    # baseline MQ-3
    ################################
    def _calibrate_mq3_start(self):
        def worker():
            self.mq3.calibrate_baseline()
            QtCore.QMetaObject.invokeMethod(
                self,
                "_baseline_done",
                QtCore.Qt.QueuedConnection
            )
        threading.Thread(target=worker, daemon=True).start()

    @Slot()
    def _baseline_done(self):
        self.set_message(now_str(), "Podejdź bliżej")

    ################################
    # zamykanie
    ################################
    def closeEvent(self, e: QtGui.QCloseEvent):
        for t in [getattr(self, "measure_timer", None),
                  getattr(self, "face_timer",    None),
                  getattr(self, "ui_timer",      None),
                  getattr(self, "cam_timer",     None)]:
            try:
                if t:
                    t.stop()
            except Exception:
                pass
        try:
            self.cam.stop()
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


############################
# Qt ENV SETUP
############################
def setup_qt_env():
    """
    Żeby nie musieć ręcznie pisać:
      export DISPLAY=:0
      export XDG_RUNTIME_DIR=/run/user/$(id -u)
      ...
    Robimy to tutaj automatycznie.
    """
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


############################
# MAIN
############################
def main():
    setup_qt_env()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()

    if CONFIG["fullscreen"]:
        win.showFullScreen()
    else:
        win.resize(CONFIG["screen_width"], CONFIG["screen_height"])
        win.show()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
