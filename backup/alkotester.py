#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alkotester – Raspberry Pi 4 kiosk (DSI 720x1280)

- Pełnoekranowa aplikacja na panelu dotykowym 720x1280 (pion).
- Podgląd kamery Picamera2:
    • bierzemy klatkę 1280x720 (landscape),
    • opcjonalnie obracamy (np. "cw" = 90° w prawo → pion),
    • cropujemy środek i skalujemy tak, żeby WYPEŁNIĆ obszar podglądu,
      bez czarnych pasów i bez rozciągania ludzi (czyli crop + scale fill).
- Kolory poprawne: używamy BGR jako "prawdy", przed renderem konwertujemy do RGB.
- Ramka twarzy + procent dopasowania w prawym dolnym rogu tej ramki.
- Logika stanów (FSM):

    INIT (kalibracja MQ-3 baseline) →
    IDLE →
    DETECT →
    (rozpoznano twarz?) → IDENTIFIED_WAIT (5 s odliczanie powitania)
                        → CALIBRATE (3 s ustaw się bliżej)
                        → MEASURE ("A dmuchnij no…" 3 s)
                        → DECIDE:
                            - PASS → otwarcie bramki, log, powrót do IDLE
                            - RETRY → przyciski [Ponów pomiar] [Odmowa]
                            - DENY  → odmowa, log, powrót do IDLE

    Jeśli nie rozpoznaliśmy twarzy:
        DETECT po ~5 próbach błędnej identyfikacji wymusza PIN_ENTRY:
            PIN_ENTRY
                - użytkownik wpisuje PIN (musi istnieć w bazie już wcześniej)
                - zbieramy 3 zdjęcia twarzy tej osoby
                - trenujemy bazę twarzy
                - przechodzimy do DETECT_RETRY
        DETECT_RETRY (max 3 próby)
            - sprawdzamy czy system już kojarzy twarz z tym PIN-em
            - jeśli tak → IDENTIFIED_WAIT
            - jeśli nie po limicie → fallback_pin_flag=True → IDENTIFIED_WAIT
              (czyli wejdzie dalej "na PIN", bo to ścieżka awaryjna)

    CALIBRATE:
        - 3-sekundowe odliczanie "Ustaw się prosto, podejdź bliżej".
        - Na koniec sprawdzamy, czy twarz jest wystarczająco duża.
        - Jeśli OK → MEASURE,
          jeśli nie OK → wracamy do DETECT (od zera).
          Jeśli fallback_pin_flag=True pomijamy ten warunek i wchodzimy w MEASURE.

- PIN fallback:
    • Użytkownik NIE dodaje nowego pracownika. PIN musi już istnieć w bazie.
    • Jeśli zły PIN → "Zły PIN – brak danych" i wracamy do IDLE.

- MCP3008 + MQ-3: pseudo-promile, zapis historii i impuls GPIO18 przy wejściu.
- Logi CSV:
    logs/events.csv        datetime;event;employee_name;employee_id
    logs/measurements.csv  datetime;employee_name;employee_id;promille;fallback_pin

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


#################################
# KONFIG
#################################
CONFIG = {
    # fizyczna rozdziałka panelu DSI (pionowo)
    "screen_width": 720,
    "screen_height": 1280,

    # pasek z komunikatem + przyciskami na dole
    "overlay_height_px": 220,

    # tryb kiosk
    "fullscreen": True,
    "kiosk_window_flags": False,   # frameless/topmost; False bo robiło małe okno w rogu
    "hide_cursor": False,          # na produkcji True żeby ukryć mysz

    # Kamera:
    # capture w landscape (1280x720),
    # rotate_dir potem obraca numpy:
    #   "cw"   = 90° w prawo (portret)
    #   "ccw"  = 90° w lewo
    #   "180"  = do góry nogami
    #   "none" = bez obrotu (poziomo)
    "camera_main_size": (1280, 720),  # (W,H) z sensora
    "camera_fps": 30,
    "rotate_dir": "cw",

    # rozpoznawanie twarzy
    "face_detect_interval_ms": 1000,  # jak często detekcja/ID
    "face_min_size": 120,
    "recognition_conf_ok": 55.0,   # %
    "recognition_conf_low": 20.0,  # %

    # ile prób w DETECT zanim wymusimy PIN
    "detect_fail_limit": 5,
    # ile prób w DETECT_RETRY zanim uznamy fallback_pin_flag
    "detect_retry_limit": 3,

    # MQ-3 / MCP3008
    "spi_bus": 0,
    "spi_device": 0,          # CE0
    "mq3_channel": 0,
    "baseline_samples": 150,
    "promille_scale": 220.0,
    "measure_seconds": 3.0,

    # progi decyzji [‰]
    "threshold_pass": 0.00,
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

    # pracownik testowy żeby baza nie była pusta
    "bootstrap_employee": {
        "id": "1",
        "name": "Kamil Karolak",
        "pin": "0000",
    },

    "debug": False,
}


#################################
# UTIL: katalogi / logi / czas
#################################
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


#################################
# MCP3008 / MQ-3
#################################
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


#################################
# FaceDB (Haar + ORB)
#################################
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
        self.emp_by_pin = {
            e["pin"]: e for e in self.employees.get("employees", []) if "pin" in e
        }
        self.emp_by_id  = {
            e.get("id") or e.get("name"): e for e in self.employees.get("employees", [])
        }

    def save_employees(self):
        with open(self.employees_json, "w", encoding="utf-8") as f:
            json.dump(self.employees, f, ensure_ascii=False, indent=2)
        self._load_employees()

    def ensure_employee_exists(self, emp_id: str, name: str, pin: str):
        """
        Upewnia się, że w employees.json istnieje rekord (nie dodajemy nowych z
        palca podczas PIN fallbacku, tylko bootstrap/testowy + prekonfigurowani).
        """
        found = False
        for e in self.employees["employees"]:
            if e.get("id") == emp_id:
                found = True
                break
        if not found:
            self.employees["employees"].append({"id": emp_id, "name": name, "pin": pin})
            self.save_employees()
        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    def add_three_shots(self, emp_id: str, imgs_bgr_list):
        """
        Dopisuje 3 zdjęcia twarzy pracownika emp_id (bez tworzenia nowego pracownika).
        """
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        for img in imgs_bgr_list:
            outp = os.path.join(folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

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
        """
        Przelatuje po wszystkich pracownikach, generuje deskryptory ORB ich twarzy.
        """
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

    def recognize_face(self, img_bgr):
        """
        img_bgr: klatka już po obrocie, tak jak użytkownik ją widzi.

        Zwraca:
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

        # nazwa do wyświetlenia
        display_name = None
        if best_emp and conf > 0:
            e = self.emp_by_id.get(best_emp)
            display_name = e.get("name") if e else best_emp

        return (best_emp if conf > 0 else None), display_name, conf, (x, y, w, h)


#################################
# PIN keypad dialog (z krzyżykiem X)
#################################
class KeypadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Wprowadź PIN"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: rgba(0,0,0,210); color: white;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16,16,16,16)
        layout.setSpacing(12)

        # górny pasek z tytułem i X
        topbar = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(title)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("font-size:28px; font-weight:600; color:white;")

        btn_close = QtWidgets.QPushButton("X")
        btn_close.setFixedSize(48,48)
        btn_close.setStyleSheet(
            "font-size:24px; font-weight:700; border-radius:12px; "
            "background:#550000; color:white;"
        )
        btn_close.clicked.connect(self.reject)

        topbar.addWidget(lbl, 1)
        topbar.addWidget(btn_close, 0, QtCore.Qt.AlignRight)
        layout.addLayout(topbar)

        # pole PIN
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
        self.edit.setFixedHeight(60)
        self.edit.setStyleSheet(
            "font-size:32px; padding:8px; border-radius:12px; "
            "background:#222; color:white;"
        )
        layout.addWidget(self.edit)

        # klawiatura num
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
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


#################################
# Camera manager
#################################
class CameraManager:
    """
    Picamera2.capture_array("main") daje RGB888.
    U nas zdarzało się, że kanały wyglądały jak BGR (mocno niebieska poświata).
    Żeby to obejść:
      - całą logikę wewnętrzną prowadzimy jako BGR (traktujemy bufor jak BGR),
      - tuż przed wyświetleniem konwertujemy cvtColor(BGR->RGB).
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
        # traktujemy frame jako BGR od tej chwili
        return frame

    def stop(self):
        try:
            self.picam.stop()
        except Exception:
            pass


#################################
# MainWindow z FSM
#################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        # GPIO setup
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
        if CONFIG["kiosk_window_flags"]:
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.X11BypassWindowManagerHint
            )
        if CONFIG["hide_cursor"]:
            self.setCursor(QtCore.Qt.BlankCursor)

        # --- GŁÓWNY LAYOUT ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0,0,0,0)
        outer.setSpacing(0)

        # podgląd kamery
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background:black;")
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        outer.addWidget(self.view, 1)

        # overlay (pasek na dole)
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

        outer.addWidget(self.overlay, 0)

        # ========== RUNTIME STATE ==========
        self.state = "INIT"
        self.current_emp_id = None
        self.current_emp_name = None
        self.fallback_pin_flag = False  # True: przechodzimy dalej po PIN bo twarz nie złapała
        self.last_face_bbox = None
        self.last_confidence = 0.0
        self.last_promille = 0.0

        self.frame_last_bgr = None

        # DETECT / RETRY liczniki
        self.detect_fail_count = 0
        self.detect_retry_count = 0

        # countdowny
        self.identified_seconds_left = 0
        self.calibrate_seconds_left = 0

        # MQ-3 pomiar
        self.measure_deadline = 0.0
        self.measure_samples = []

        # post training action
        self.post_training_action = None

        # Kamera manager
        self.cam = CameraManager(
            CONFIG["camera_main_size"][0],
            CONFIG["camera_main_size"][1],
            CONFIG["rotate_dir"],
        )

        # Timery
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self.on_camera_tick)
        self.cam_timer.start(int(1000/max(1,CONFIG["camera_fps"])))

        self.face_timer = QtCore.QTimer(self)
        self.face_timer.timeout.connect(self.on_face_tick)
        # start/stop w zależności od stanu

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.on_ui_tick)
        # start/stop w zależności od stanu

        self.identified_timer = QtCore.QTimer(self)
        self.identified_timer.timeout.connect(self.on_identified_tick)

        self.calibrate_timer = QtCore.QTimer(self)
        self.calibrate_timer.timeout.connect(self.on_calibrate_tick)

        self.measure_timer = QtCore.QTimer(self)
        self.measure_timer.timeout.connect(self.on_measure_tick)

        # Przyciski
        self.btn_primary.clicked.connect(self.on_btn_primary)
        self.btn_secondary.clicked.connect(self.on_btn_secondary)

        # Startowy komunikat – baseline MQ-3
        self.set_message(
            "Proszę czekać…",
            "Kalibracja czujnika MQ-3 w toku",
            color="white",
        )
        self.show_buttons(primary_text=None, secondary_text=None)

        # baseline MQ3 w wątku, po nim wchodzimy do IDLE
        self._calibrate_mq3_start()

    #################################
    # --- TIMER HELPERS ---
    #################################
    def _stop_timer(self, t: QtCore.QTimer):
        try:
            if t.isActive():
                t.stop()
        except Exception:
            pass

    #################################
    # --- UI helpers ---
    #################################
    def set_message(self, top, center=None, color="white"):
        """
        Ustawia oba labelki naraz + kolor (white/green/red).
        """
        if color == "green":
            c = "#00ff00"
        elif color == "red":
            c = "#ff4444"
        else:
            c = "white"

        self.lbl_top.setText(top)
        self.lbl_top.setStyleSheet(
            f"color:{c}; font-size:28px; font-weight:600;"
        )

        self.lbl_center.setText(center or "")
        self.lbl_center.setStyleSheet(
            f"color:{c}; font-size:36px; font-weight:700;"
        )

    def show_buttons(self, primary_text=None, secondary_text=None):
        if primary_text is None:
            self.btn_primary.hide()
        else:
            self.btn_primary.setText(primary_text)
            self.btn_primary.show()

        if secondary_text is None:
            self.btn_secondary.hide()
        else:
            self.btn_secondary.setText(secondary_text)
            self.btn_secondary.show()

    #################################
    # --- FSM ENTER STATE FUNCTIONS ---
    #################################
    def enter_idle(self):
        """
        IDLE:
          - kamera leci
          - czekamy aż ktoś podejdzie
          - co sekundę face_rec
          - przycisk "Wprowadź PIN"
        """
        self.state = "IDLE"
        self.current_emp_id = None
        self.current_emp_name = None
        self.fallback_pin_flag = False
        self.last_face_bbox = None
        self.last_confidence = 0.0
        self.detect_fail_count = 0
        self.detect_retry_count = 0
        self.identified_seconds_left = 0
        self.calibrate_seconds_left = 0
        self.measure_deadline = 0.0
        self.measure_samples = []

        # timery
        self.ui_timer.start(250)
        self.face_timer.start(CONFIG["face_detect_interval_ms"])
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)

        self.set_message(now_str(), "Podejdź bliżej", color="white")
        self.show_buttons(primary_text=None, secondary_text="Wprowadź PIN")

    def enter_detect(self):
        """
        DETECT:
          - ktoś stoi, próbujemy rozpoznać
          - po kilku próbach niepowodzenia -> PIN_ENTRY
        """
        self.state = "DETECT"
        self.detect_fail_count = 0

        self.ui_timer.start(250)
        self.face_timer.start(CONFIG["face_detect_interval_ms"])
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)

        self.set_message(now_str(), "Szukam twarzy…", color="white")
        self.show_buttons(primary_text=None, secondary_text="Wprowadź PIN")

    def enter_pin_entry(self):
        """
        PIN_ENTRY:
          - zatrzymujemy face_timer żeby nic nie mieszało
          - pokazujemy KeypadDialog
        """
        self.state = "PIN_ENTRY"
        self._stop_timer(self.face_timer)
        self._stop_timer(self.ui_timer)
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)

        dlg = KeypadDialog(self, title="Wprowadź PIN")
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            pin = dlg.value()
            emp = self.facedb.emp_by_pin.get(pin)
            if not emp:
                # zły PIN
                self.set_message("Zły PIN – brak danych", "", color="red")
                self.show_buttons(primary_text=None, secondary_text=None)
                QtCore.QTimer.singleShot(2000, self.enter_idle)
                return

            # poprawny PIN
            self.current_emp_id = emp.get("id") or emp.get("name")
            self.current_emp_name = emp.get("name")
            self.fallback_pin_flag = False  # jeszcze nie awaryjnie, spróbujemy retrainingu

            # zbierz 3 ujęcia twarzy i potem trening
            self.collect_new_shots_for_current_emp()
        else:
            # anulowano krzyżykiem
            self.enter_idle()

    def enter_detect_retry(self):
        """
        DETECT_RETRY:
          - po PIN + treningu: sprawdzamy jeszcze raz,
            czy kamera rozpozna twarz TEGO KONKRETNEGO pracownika.
          - jeśli nadal lipa po detect_retry_limit próbach -> fallback_pin_flag=True
        """
        self.state = "DETECT_RETRY"
        self.detect_retry_count = 0

        self.face_timer.start(CONFIG["face_detect_interval_ms"])
        self._stop_timer(self.ui_timer)
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)

        self.set_message("Sprawdzam twarz…", self.current_emp_name or "", color="white")
        self.show_buttons(primary_text=None, secondary_text=None)

    def enter_identified_wait(self):
        """
        IDENTIFIED_WAIT:
          - Mamy current_emp_id/current_emp_name przypisane.
          - Pokazujemy powitanie i odliczamy 5s, potem CALIBRATE.
        """
        self.state = "IDENTIFIED_WAIT"
        self.identified_seconds_left = 5

        self._stop_timer(self.face_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)
        self._stop_timer(self.ui_timer)

        self.identified_timer.start(1000)

        top  = f"Cześć {self.current_emp_name or ''}"
        cent = f"Za {self.identified_seconds_left} s zaczynamy pomiar"
        self.set_message(top, cent, color="white")
        self.show_buttons(primary_text=None, secondary_text=None)

    def enter_calibrate(self):
        """
        CALIBRATE:
          - prosimy "Ustaw się prosto, podejdź bliżej"
          - 3s odliczania
          - w tym stanie znowu działa face_timer żeby mieć bbox do oceny
        """
        self.state = "CALIBRATE"
        self.calibrate_seconds_left = 3

        self.face_timer.start(CONFIG["face_detect_interval_ms"])
        self._stop_timer(self.ui_timer)
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.measure_timer)

        self.calibrate_timer.start(1000)

        self.set_message(
            "Ustaw się prosto, podejdź bliżej",
            f"Start za {self.calibrate_seconds_left} s",
            color="white",
        )
        self.show_buttons(primary_text=None, secondary_text=None)

    def enter_measure(self):
        """
        MEASURE:
          - zbieramy próbki MQ-3 przez ~3s
          - brak face_timer (nie chcemy już nic ruszać)
        """
        self.state = "MEASURE"
        self.measure_samples = []
        self.measure_deadline = time.time() + CONFIG["measure_seconds"]

        self._stop_timer(self.face_timer)
        self._stop_timer(self.ui_timer)
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)

        self.measure_timer.start(100)

        self.set_message(
            "A dmuchnij no…",
            f"{CONFIG['measure_seconds']:.1f} s",
            color="white",
        )
        self.show_buttons(primary_text=None, secondary_text=None)

    def enter_decide(self, promille):
        """
        DECIDE:
          - sprawdzamy progi
          - PASS: zielony, otwórz przejście, log, powrót po chwili do IDLE
          - RETRY: czerwony, pokaż przyciski "Ponów pomiar" / "Odmowa"
          - DENY: czerwony, "Odmowa", log, powrót po chwili do IDLE
        """
        self.last_promille = promille
        prom_s = f"Pomiar: {promille:.3f} [‰]"

        self._stop_timer(self.face_timer)
        self._stop_timer(self.ui_timer)
        self._stop_timer(self.identified_timer)
        self._stop_timer(self.calibrate_timer)
        self._stop_timer(self.measure_timer)

        # PASS
        if promille <= CONFIG["threshold_pass"]:
            self.state = "DECIDE_PASS"
            self.set_message(prom_s, "Przejście otwarte", color="green")
            self.show_buttons(primary_text=None, secondary_text=None)
            self.trigger_gate_and_log(True, promille)
            QtCore.QTimer.singleShot(2500, self.enter_idle)
            return

        # RETRY region
        if promille < CONFIG["threshold_deny"]:
            self.state = "RETRY"
            self.set_message(
                prom_s,
                "Ponów pomiar",
                color="red",
            )
            self.show_buttons(primary_text="Ponów pomiar", secondary_text="Odmowa")
            return

        # DENY
        self.state = "DECIDE_DENY"
        self.set_message(prom_s, "Odmowa", color="red")
        self.show_buttons(primary_text=None, secondary_text=None)
        self.trigger_gate_and_log(False, promille)
        QtCore.QTimer.singleShot(3000, self.enter_idle)

    #################################
    # --- Countdown tick handlers ---
    #################################
    def on_identified_tick(self):
        if self.state != "IDENTIFIED_WAIT":
            self._stop_timer(self.identified_timer)
            return

        self.identified_seconds_left -= 1
        if self.identified_seconds_left > 0:
            top  = f"Cześć {self.current_emp_name or ''}"
            cent = f"Za {self.identified_seconds_left} s zaczynamy pomiar"
            self.set_message(top, cent, color="white")
        else:
            self._stop_timer(self.identified_timer)
            self.enter_calibrate()

    def on_calibrate_tick(self):
        if self.state != "CALIBRATE":
            self._stop_timer(self.calibrate_timer)
            return

        self.calibrate_seconds_left -= 1
        if self.calibrate_seconds_left > 0:
            self.set_message(
                "Ustaw się prosto, podejdź bliżej",
                f"Start za {self.calibrate_seconds_left} s",
                color="white",
            )
        else:
            self._stop_timer(self.calibrate_timer)
            # po odliczeniu decydujemy czy twarz jest blisko
            if self.fallback_pin_flag:
                # awaryjnie po PIN – przechodzimy mimo wszystko
                self.enter_measure()
                return

            ok = False
            if self.last_face_bbox is not None:
                (_, _, w, h) = self.last_face_bbox
                if max(w, h) >= CONFIG["face_min_size"]:
                    ok = True

            if ok:
                self.enter_measure()
            else:
                # twarz za daleko -> wracamy do DETECT
                self.enter_detect()

    def on_measure_tick(self):
        if self.state != "MEASURE":
            self._stop_timer(self.measure_timer)
            return

        left = self.measure_deadline - time.time()
        self.measure_samples.append(self.mq3.read_raw())

        if left > 0:
            self.set_message(
                "A dmuchnij no…",
                f"{left:0.1f} s",
                color="white",
            )
        else:
            self._stop_timer(self.measure_timer)
            promille = self.mq3.promille_from_samples(self.measure_samples)
            self.enter_decide(promille)

    #################################
    # --- Przyciski overlay ---
    #################################
    def on_btn_primary(self):
        # primary = "Ponów pomiar" (tylko w stanie RETRY)
        if self.state == "RETRY":
            self.enter_measure()

    def on_btn_secondary(self):
        # secondary:
        #   - w RETRY = "Odmowa"
        #   - w IDLE/DETECT = "Wprowadź PIN"
        if self.state == "RETRY":
            # odmowa -> log odmowy i powrót do IDLE
            self.set_message("Odmowa", "", color="red")
            self.trigger_gate_and_log(False, self.last_promille)
            self.show_buttons(primary_text=None, secondary_text=None)
            QtCore.QTimer.singleShot(2000, self.enter_idle)
            return

        if self.state in ("IDLE", "DETECT"):
            self.enter_pin_entry()

    #################################
    # --- PIN capture & training ---
    #################################
    def collect_new_shots_for_current_emp(self):
        """
        Robimy 3 snapshoty z kamery dla bieżącego self.current_emp_id,
        zapisujemy do faces/<id>/ i reindeksujemy bazę.
        Potem -> DETECT_RETRY
        """
        emp_id = self.current_emp_id
        if not emp_id:
            # safety, wróć do IDLE
            self.enter_idle()
            return

        self.set_message(
            "Przytrzymaj twarz w kadrze",
            "Zapisuję zdjęcia…",
            color="white",
        )
        self.show_buttons(primary_text=None, secondary_text=None)

        imgs = []

        def snap(i):
            if self.frame_last_bgr is not None:
                imgs.append(self.frame_last_bgr.copy())
            if i < 3:
                QtCore.QTimer.singleShot(800, lambda: snap(i+1))
            else:
                # mamy 3 zdjęcia
                self.facedb.add_three_shots(emp_id, imgs)
                self.training_start(post_action="DETECT_RETRY")

        snap(1)

    def training_start(self, post_action):
        """
        Reindeksuj ORB w tle, a po skończeniu -> post_action
        """
        self.post_training_action = post_action

        self.set_message("Proszę czekać…", "Trening AI", color="white")
        self.show_buttons(primary_text=None, secondary_text=None)

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
        act = self.post_training_action
        self.post_training_action = None

        if act == "DETECT_RETRY":
            self.enter_detect_retry()
        else:
            # fallback: wróć do DETECT
            self.enter_detect()

    #################################
    # --- BRAMKA + LOGI ---
    #################################
    def trigger_gate_and_log(self, pass_ok: bool, promille: float):
        emp_name = self.current_emp_name or "<nieznany>"
        emp_id   = self.current_emp_id or "<none>"
        ts = datetime.now().isoformat()

        if pass_ok:
            # otwórz przekaźnik
            GPIO.output(CONFIG["gate_gpio"], GPIO.HIGH)

            def pulse():
                time.sleep(CONFIG["gate_pulse_sec"])
                GPIO.output(CONFIG["gate_gpio"], GPIO.LOW)

            threading.Thread(target=pulse, daemon=True).start()

            log_csv(
                os.path.join(CONFIG["logs_dir"], "events.csv"),
                ["datetime","event","employee_name","employee_id"],
                [ts,"gate_open",emp_name,emp_id]
            )
        else:
            log_csv(
                os.path.join(CONFIG["logs_dir"], "events.csv"),
                ["datetime","event","employee_name","employee_id"],
                [ts,"deny_access",emp_name,emp_id]
            )

        log_csv(
            os.path.join(CONFIG["logs_dir"], "measurements.csv"),
            ["datetime","employee_name","employee_id","promille","fallback_pin"],
            [ts,emp_name,emp_id,f"{promille:.3f}",int(self.fallback_pin_flag)]
        )

    #################################
    # --- Preview helpers ---
    #################################
    def _crop_and_scale_fill(self, src_rgb, target_w, target_h):
        """
        Wypełnij cały widget bez czarnych pasów:
          1. przytnij środek tak, by aspect == target_w/target_h,
          2. przeskaluj do dokładnie (target_w, target_h).
        """
        if target_w <= 0 or target_h <= 0:
            return None

        sh, sw, _ = src_rgb.shape
        target_aspect = target_w / float(target_h)
        src_aspect = sw / float(sh)

        if src_aspect > target_aspect:
            # obraz "za szeroki" → przytnij szerokość
            new_sw = int(target_aspect * sh)
            if new_sw > sw:
                new_sw = sw
            x0 = (sw - new_sw) // 2
            cropped = src_rgb[:, x0:x0+new_sw, :]
        else:
            # obraz "za wysoki/wąski" → przytnij wysokość
            new_sh = int(sw / target_aspect)
            if new_sh > sh:
                new_sh = sh
            y0 = (sh - new_sh) // 2
            cropped = src_rgb[y0:y0+new_sh, :, :]

        fitted = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return fitted

    #################################
    # --- CAMERA TICK (30fps approx) ---
    #################################
    def on_camera_tick(self):
        frame_bgr = self.cam.get_frame_bgr()
        if frame_bgr is None:
            return

        # zapamiętujemy ostatnią klatkę w BGR
        self.frame_last_bgr = frame_bgr.copy()

        # dorysuj ramkę twarzy + % dopasowania
        disp_bgr = frame_bgr.copy()

        if self.last_face_bbox is not None:
            (x, y, w, h) = self.last_face_bbox
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

            # kolor ramki zależnie od pewności
            if self.last_confidence >= CONFIG["recognition_conf_ok"]:
                color = (0,255,0)        # zielony
            elif self.last_confidence <= CONFIG["recognition_conf_low"]:
                color = (0,255,255)      # żółty
            else:
                color = (255,255,0)      # cyjan-ish

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

        # BGR -> RGB do Qt
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

        # dopasuj do aktualnego rozmiaru widżetu self.view
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

    #################################
    # --- FACE TICK (co 1 s) ---
    #################################
    def on_face_tick(self):
        if self.frame_last_bgr is None:
            return

        # detekcja twarzy / identyfikacja
        emp_id, emp_name, conf, bbox = self.facedb.recognize_face(self.frame_last_bgr)

        # aktualizuj bbox i pewność do wyświetlania
        self.last_face_bbox = bbox
        self.last_confidence = conf or 0.0

        # logika zależna od stanu
        if self.state == "IDLE":
            # jeżeli ktoś jest w kadrze -> przechodzimy do DETECT
            if bbox is not None:
                self.enter_detect()
            return

        if self.state == "DETECT":
            if bbox is None:
                # nikt nie stoi -> resetuj licznik, komunikat "Szukam twarzy…"
                self.detect_fail_count = 0
                self.set_message(now_str(), "Szukam twarzy…", color="white")
                return

            if emp_name and conf >= CONFIG["recognition_conf_ok"]:
                # Rozpoznano pewnie -> przypisz użytkownika i idziemy dalej
                self.current_emp_id = emp_id
                self.current_emp_name = emp_name
                self.fallback_pin_flag = False
                self.enter_identified_wait()
                return

            # Niepewne rozpoznanie
            self.detect_fail_count += 1
            if self.detect_fail_count >= CONFIG["detect_fail_limit"]:
                # przechodzimy do PIN
                self.enter_pin_entry()
                return

            # pokazuj feedback
            if conf <= CONFIG["recognition_conf_low"]:
                self.set_message(now_str(), "Nie rozpoznaję…", color="white")
            else:
                self.set_message(now_str(), f"pewność: {conf:.0f}%", color="white")
            return

        if self.state == "DETECT_RETRY":
            # Po PIN + treningu
            self.detect_retry_count += 1

            if emp_id == self.current_emp_id and conf >= CONFIG["recognition_conf_ok"]:
                # udało się dopasować nowo wytrenowaną twarz
                self.fallback_pin_flag = False
                self.enter_identified_wait()
                return

            if self.detect_retry_count >= CONFIG["detect_retry_limit"]:
                # nadal lipa -> fallback_pin_flag True i jedziemy dalej
                self.fallback_pin_flag = True
                self.enter_identified_wait()
                return

            # wciąż próbujemy, aktualizuj komunikat
            txt_conf = f"{conf:.0f}%" if conf is not None else ""
            self.set_message(
                "Sprawdzam twarz…",
                f"{self.current_emp_name or ''} {txt_conf}",
                color="white",
            )
            return

        if self.state == "CALIBRATE":
            # podczas CALIBRATE zbieramy bbox żeby ocenić dystans,
            # ale nie zmieniamy stanu tutaj
            return

        # w pozostałych stanach face_tick nic nie robi

    #################################
    # --- UI tick (odświeża zegar) ---
    #################################
    def on_ui_tick(self):
        if self.state in ("IDLE","DETECT"):
            # aktualizuj tylko górny label (czas), nie zmieniaj koloru
            c = "white"
            self.lbl_top.setText(now_str())
            self.lbl_top.setStyleSheet(
                f"color:{c}; font-size:28px; font-weight:600;"
            )
        # inne stany nie aktualizują zegara

    #################################
    # --- baseline MQ-3 ---
    #################################
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
        # baseline gotowy => wchodzimy do IDLE
        self.enter_idle()

    #################################
    # --- zamykanie ---
    #################################
    def closeEvent(self, e: QtGui.QCloseEvent):
        # zatrzymaj timery
        for t in [
            getattr(self, "measure_timer", None),
            getattr(self, "calibrate_timer", None),
            getattr(self, "identified_timer", None),
            getattr(self, "face_timer", None),
            getattr(self, "ui_timer", None),
            getattr(self, "cam_timer", None),
        ]:
            try:
                if t and t.isActive():
                    t.stop()
            except Exception:
                pass

        # zatrzymaj kamerę
        try:
            self.cam.stop()
        except Exception:
            pass

        # zamknij SPI
        try:
            self.adc.close()
        except Exception:
            pass

        try:
            GPIO.cleanup()
        except Exception:
            pass

        return super().closeEvent(e)


#################################
# Qt ENV SETUP
#################################
def setup_qt_env():
    """
    Żeby nie musieć ręcznie pisać:
      export DISPLAY=:0
      export XDG_RUNTIME_DIR=/run/user/$(id -u)
      export QT_QPA_PLATFORM=xcb
      ...
    Automatycznie ustawiamy środowisko dla renderowania na lokalnym DSI.
    """
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


#################################
# MAIN
#################################
def main():
    setup_qt_env()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()

    if CONFIG["fullscreen"]:
        win.showFullScreen()
    else:
        win.resize(CONFIG["screen_width"], CONFIG["screen_height"])
        win.show()

    # obsługa Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
