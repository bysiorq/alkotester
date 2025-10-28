#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py – Główna aplikacja (FSM, UI, logika pomiaru, otwieranie bramki)

Stan maszyny:
INIT → IDLE → DETECT → IDENTIFIED_WAIT → CALIBRATE → MEASURE → DECIDE

Funkcje:
- UI i timery
- rozpoznawanie twarzy (FaceDB)
- fallback PIN
- pomiar MQ-3
- logi + przekaźnik

Uruchamianie:
    python3 main.py
"""

import os
import sys
import cv2
import time
import signal
import threading
import numpy as np
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
import RPi.GPIO as GPIO

from config import CONFIG
from utils_fs import ensure_dirs, now_str, log_csv
from sensors import MCP3008, MQ3Sensor
from facedb import FaceDB
from camera_manager import CameraManager
from keypad import KeypadDialog


#################################
# jakość pojedynczej próbki twarzy
#################################
def _face_quality(gray_roi):
    """
    Zwraca (ok, sharpness, brightness).
    - sharpness: wariancja Laplace'a (im większa, tym ostrzej)
    - brightness: średnia jasność 0..255
    """
    sharp = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    bright = float(np.mean(gray_roi))
    ok = (
        sharp >= CONFIG["quality_min_sharpness"]
        and CONFIG["quality_min_brightness"] <= bright <= CONFIG["quality_max_brightness"]
    )
    return ok, sharp, bright


#################################
# GŁÓWNE OKNO / FSM
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

        # Okno kiosk / wygląd
        self.setWindowTitle("Alkotester – Raspberry Pi")
        if CONFIG["kiosk_window_flags"]:
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.X11BypassWindowManagerHint
            )
        if CONFIG["hide_cursor"]:
            self.setCursor(QtCore.Qt.BlankCursor)

        # --- LAYOUT ekranu ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Podgląd kamery (góra)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background:black;")
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        outer.addWidget(self.view, 1)

        # Pasek overlay (dół)
        self.overlay = QtWidgets.QFrame()
        self.overlay.setFixedHeight(CONFIG["overlay_height_px"])
        self.overlay.setStyleSheet("background: rgba(0,0,0,110); color:white;")

        ov = QtWidgets.QVBoxLayout(self.overlay)
        ov.setContentsMargins(16, 12, 16, 12)
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

        # ========== STAN RUNTIME ==========
        self.state = "INIT"

        self.current_emp_id = None
        self.current_emp_name = None

        # fallback_pin_flag=True oznacza, że dopuścimy pomiar/wyjście dalej
        # nawet jeśli twarz nie została poprawnie złapana (awaria rozpoznania).
        self.fallback_pin_flag = False

        self.last_face_bbox = None
        self.last_confidence = 0.0
        self.last_promille = 0.0

        self.frame_last_bgr = None

        # DETECT/RETRY liczniki
        self.detect_fail_count = 0
        self.detect_retry_count = 0

        # stabilizacja rozpoznania (żeby nie brać kogoś po jednym ticku)
        self._stable_emp_id = None
        self._stable_count = 0

        # countdowny
        self.identified_seconds_left = 0
        self.calibrate_seconds_left = 0

        # w CALIBRATE zbieramy:
        # - czy twarz była kiedykolwiek duża/blisko (good_face)
        # - czy w ogóle kogoś widzieliśmy (seen_face)
        self.calibrate_good_face = False
        self.calibrate_seen_face = False

        # MQ-3 pomiar
        self.measure_deadline = 0.0
        self.measure_samples = []

        # po treningu kogo mamy robić dalej
        self.post_training_action = None

        # Kamera
        self.cam = CameraManager(
            CONFIG["camera_main_size"][0],
            CONFIG["camera_main_size"][1],
            CONFIG["rotate_dir"],
        )

        # Timery
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self.on_camera_tick)
        self.cam_timer.start(int(1000 / max(1, CONFIG["camera_fps"])))

        self.face_timer = QtCore.QTimer(self)
        self.face_timer.timeout.connect(self.on_face_tick)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.on_ui_tick)

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

        # Kalibracja MQ-3 baseline w wątku, po niej -> IDLE
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
          - co sekundę face_rec (szukanie twarzy)
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
        self._stable_emp_id = None
        self._stable_count = 0

        self.identified_seconds_left = 0
        self.calibrate_seconds_left = 0
        self.calibrate_good_face = False
        self.calibrate_seen_face = False

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
          - ktoś stoi, próbujemy rozpoznać (face_tick)
          - po kilku próbach niepowodzenia -> PIN_ENTRY
        """
        self.state = "DETECT"
        self.detect_fail_count = 0
        self._stable_emp_id = None
        self._stable_count = 0

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
          - pokazujemy KeypadDialog z krzyżykiem
          - tylko weryfikujemy PIN istniejącego pracownika
          - jeśli PIN ok -> zbieramy próbki twarzy tego pracownika
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
            self.fallback_pin_flag = False  # jeszcze nie awaryjnie

            # zbierz DOBRE próbki twarzy i potem trenuj
            self.collect_new_shots_for_current_emp()
        else:
            # anulowano krzyżykiem
            self.enter_idle()


    def enter_detect_retry(self):
        """
        DETECT_RETRY:
          - Po PIN + treningu: sprawdzamy czy kamera rozpozna
            TEGO KONKRETNEGO pracownika.
          - Jeśli wciąż nie łapie po detect_retry_limit → fallback_pin_flag=True,
            i mimo wszystko idziemy dalej.
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
          - Mamy current_emp_id/current_emp_name.
          - Pokazujemy powitanie i odliczamy 5 s, potem CALIBRATE.
          - Ramka twarzy nie jest już potrzebna → chowamy.
        """
        self.state = "IDENTIFIED_WAIT"
        self.identified_seconds_left = 5

        # usuń starą ramkę żeby nie wisiała zamrożona
        self.last_face_bbox = None
        self.last_confidence = 0.0

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
          - 3 s odliczania
          - w tym stanie znowu działa face_timer, żeby mieć bbox i ocenić dystans
          - zapamiętujemy:
              self.calibrate_good_face = True  -> twarz była DUŻA/blisko
              self.calibrate_seen_face = True  -> twarz była WIDOCZNA w ogóle
        """
        self.state = "CALIBRATE"
        self.calibrate_seconds_left = 3
        self.calibrate_good_face = False
        self.calibrate_seen_face = False

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
          - zbieramy próbki MQ-3 przez ~3 s
          - brak face_timer (twarz nas już nie obchodzi)
          - ramkę twarzy chowamy
        """
        self.state = "MEASURE"
        self.measure_samples = []
        self.measure_deadline = time.time() + CONFIG["measure_seconds"]

        self.last_face_bbox = None
        self.last_confidence = 0.0

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
          - PASS: zielony, otwórz przejście, log, powrót do IDLE
          - RETRY: czerwony, przyciski "Ponów pomiar" / "Odmowa"
          - DENY: czerwony, "Odmowa", log, powrót do IDLE
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

        # RETRY
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
        """
        Licznik w stanie IDENTIFIED_WAIT:
        po 5s przechodzimy do CALIBRATE.
        """
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
        """
        Licznik w stanie CALIBRATE.
        Po 3s decydujemy:
        - czy iść do pomiaru
        - czy wrócić do DETECT (jeśli nikogo nie było)
        """
        if self.state != "CALIBRATE":
            self._stop_timer(self.calibrate_timer)
            return

        self.calibrate_seconds_left -= 1
        if self.calibrate_seconds_left > 0:
            # aktualizuj napis z odliczaniem
            self.set_message(
                "Ustaw się prosto, podejdź bliżej",
                f"Start za {self.calibrate_seconds_left} s",
                color="white",
            )
            return

        # koniec odliczania CALIBRATE
        self._stop_timer(self.calibrate_timer)

        # Priorytety przejścia do pomiaru:
        #
        # 1. fallback_pin_flag == True:
        #    -> osoba została dopuszczona awaryjnie (PIN / niepewne ID),
        #       więc i tak robimy pomiar.
        #
        # 2. calibrate_good_face == True:
        #    -> kamera widziała twarz wystarczająco DUŻĄ (blisko),
        #       więc robimy pomiar normalnie.
        #
        # 3. calibrate_seen_face == True:
        #    -> ktoś BYŁ w kadrze podczas kalibracji, ale zniknął,
        #       bo schylił się do ustnika.
        #       W takim razie nadal przechodzimy do pomiaru,
        #       ale ustawiamy fallback_pin_flag=True,
        #       żeby to było jasne w logach.
        #
        # 4. W przeciwnym razie:
        #    -> nikogo fizycznie nie było, przerwij i wróć do DETECT.

        if self.fallback_pin_flag:
            self.enter_measure()
            return

        if self.calibrate_good_face:
            self.enter_measure()
            return

        if self.calibrate_seen_face:
            self.fallback_pin_flag = True
            self.enter_measure()
            return

        # nikt nie stał przy urządzeniu podczas kalibracji → wracamy do szukania
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
        Zbieramy dobre próbki twarzy danego pracownika:
          - twarz w kadrze
          - wystarczająco duża
          - ostra i dobrze oświetlona
        Zapisujemy wycięty face ROI (240x240 BGR).

        Jeśli w train_timeout_sec nie zbierzemy train_required_shots -> anuluj.
        Po sukcesie robimy trening i przechodzimy do DETECT_RETRY.
        """
        emp_id = self.current_emp_id
        if not emp_id:
            self.enter_idle()
            return

        need = CONFIG["train_required_shots"]
        timeout_s = CONFIG["train_timeout_sec"]
        deadline = time.time() + timeout_s

        self.set_message(
            "Przytrzymaj twarz w obwódce",
            f"Zbieram próbki 0/{need}",
            color="white",
        )
        self.show_buttons(primary_text=None, secondary_text=None)

        saved = 0
        imgs = []

        def tick():
            nonlocal saved, imgs, deadline

            if time.time() > deadline:
                # timeout
                self.set_message("Nie udało się zebrać próbek", "Spróbuj ponownie", color="red")
                QtCore.QTimer.singleShot(2000, self.enter_idle)
                return

            if self.frame_last_bgr is None:
                QtCore.QTimer.singleShot(80, tick)
                return

            frame = self.frame_last_bgr
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedb.cascade.detectMultiScale(gray, 1.2, 5)

            if len(faces) == 0:
                self.set_message(
                    "Przytrzymaj twarz w obwódce",
                    f"Zbieram próbki {saved}/{need}",
                    color="white",
                )
                QtCore.QTimer.singleShot(80, tick)
                return

            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

            # Czy twarz wystarczająco duża?
            if max(w, h) < CONFIG["face_min_size"]:
                self.set_message(
                    "Podejdź bliżej",
                    f"Zbieram próbki {saved}/{need}",
                    color="white",
                )
                QtCore.QTimer.singleShot(80, tick)
                return

            # ocena jakości (ostrość/jasność)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (240, 240), interpolation=cv2.INTER_LINEAR)
            ok, sharp, bright = _face_quality(roi_gray_resized)
            if not ok:
                self.set_message(
                    "Stań prosto, światło/ostrość",
                    f"ostrość {sharp:0.0f}, jasność {bright:0.0f}  [{saved}/{need}]",
                    color="white",
                )
                QtCore.QTimer.singleShot(80, tick)
                return

            # zapisz wyciętą twarz w BGR 240x240
            face_bgr = frame[y:y+h, x:x+w].copy()
            face_bgr = cv2.resize(face_bgr, (240, 240), interpolation=cv2.INTER_LINEAR)
            imgs.append(face_bgr)
            saved += 1

            self.set_message(
                "Próbka zapisana",
                f"Zbieram próbki {saved}/{need}",
                color="green",
            )

            if saved >= need:
                # zapis do dysku + trening
                self.facedb.add_three_shots(emp_id, imgs)
                self.training_start(post_action="DETECT_RETRY")
                return

            QtCore.QTimer.singleShot(120, tick)

        # startujemy pętlę zbierania
        QtCore.QTimer.singleShot(80, tick)


    def training_start(self, post_action):
        """
        Reindeksuj bazę twarzy w tle; po zakończeniu -> post_action.
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

    @QtCore.pyqtSlot()
    def _training_done(self):
        act = self.post_training_action
        self.post_training_action = None

        if act == "DETECT_RETRY":
            self.enter_detect_retry()
        else:
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

    def _online_learn_face(self, emp_id: str):
        """
        Jeśli mamy pewne rozpoznanie (czyli wiemy że to emp_id),
        to spróbuj dodać świeżą próbkę twarzy tej osoby do bazy,
        żeby baza cały czas się aktualizowała (broda, światło itd.).

        - bierze self.last_face_bbox i self.frame_last_bgr,
        - wycina twarz, skaluje do 240x240 BGR,
        - sprawdza jakość (_face_quality),
        - jeśli OK -> self.facedb.add_online_face_sample(emp_id, face_bgr_240)

        Ta funkcja jest celowo "miękka": jeśli coś się nie uda,
        nie rozwala reszty FSM.
        """
        try:
            if self.last_face_bbox is None:
                return
            if self.frame_last_bgr is None:
                return

            (fx, fy, fw, fh) = self.last_face_bbox
            # bezpieczeństwo (konwersja na int, plus clamp żeby nie wyjść poza frame)
            fx = int(max(0, fx))
            fy = int(max(0, fy))
            fw = int(max(0, fw))
            fh = int(max(0, fh))

            h_img, w_img, _ = self.frame_last_bgr.shape
            x2 = min(fx + fw, w_img)
            y2 = min(fy + fh, h_img)
            if x2 <= fx or y2 <= fy:
                return  # bbox bez sensu

            face_bgr = self.frame_last_bgr[fy:y2, fx:x2].copy()
            face_bgr = cv2.resize(face_bgr, (240, 240), interpolation=cv2.INTER_LINEAR)

            # sprawdź jakość zanim uczymy bazę (żeby nie nauczyć się blur-u)
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            ok_q, sharp, bright = _face_quality(face_gray)
            if not ok_q:
                return

            # zapisz/ucz tę próbką (rolling buffer w FaceDB)
            self.facedb.add_online_face_sample(emp_id, face_bgr)

        except Exception:
            # Nie chcemy żeby ewentualny wyjątek zabił GUI/timer.
            pass

    #################################
    # --- CAMERA TICK (ok. 30fps) ---
    #################################
    def on_camera_tick(self):
        frame_bgr = self.cam.get_frame_bgr()
        if frame_bgr is None:
            return

        # zapamiętujemy ostatnią klatkę
        self.frame_last_bgr = frame_bgr.copy()

        # przygotuj obraz do wyświetlenia z ramką twarzy i % pewności
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
                color = (255,255,0)      # jasny

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

        # BGR -> RGB dla Qt
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

        # wypełnij widget podglądu (crop+scale fill)
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
    # --- FACE TICK (co ~1 s) ---
    #################################
    def on_face_tick(self):
        if self.frame_last_bgr is None:
            return

        # rozpoznanie twarzy (FaceDB ma anty-false-positive)
        emp_id, emp_name, conf, bbox = self.facedb.recognize_face(self.frame_last_bgr)

        # zapisz bbox + confidence do overlay (kamera_tick narysuje ramkę)
        self.last_face_bbox = bbox
        self.last_confidence = conf or 0.0

        # --- FSM logika stanu ---

        # 1) IDLE -> jeśli pojawiła się twarz, przechodzimy do DETECT
        if self.state == "IDLE":
            if bbox is not None:
                self.enter_detect()
            return

        # 2) DETECT -> próbujemy ustalić kto to stabilnie
        if self.state == "DETECT":
            if bbox is None:
                # nikt już nie stoi -> reset state'u poszukiwania
                self.detect_fail_count = 0
                self._stable_emp_id = None
                self._stable_count = 0
                self.set_message(now_str(), "Szukam twarzy…", color="white")
                return

            # stabilizacja: ten sam emp_id przez N ticków z rzędu
            target_emp = emp_id if emp_id else None
            if target_emp is not None:
                if self._stable_emp_id == target_emp:
                    self._stable_count += 1
                else:
                    self._stable_emp_id = target_emp
                    self._stable_count = 1
            else:
                self._stable_emp_id = None
                self._stable_count = 0

            # jeśli mamy stabilne rozpoznanie tej samej osoby
            if (
                emp_name and
                conf >= CONFIG["recognition_conf_ok"] and
                self._stable_emp_id == emp_id and
                self._stable_count >= CONFIG["recognition_stable_ticks"]
            ):
                # Uff, jesteśmy pewni kto to jest.
                self.current_emp_id = emp_id
                self.current_emp_name = emp_name
                self.fallback_pin_flag = False

                # spróbuj nauczyć się świeżej próbki twarzy tej osoby
                self._online_learn_face(emp_id)

                # dalej normalny flow → IDENTIFIED_WAIT
                self.enter_identified_wait()
                return

            # brak stabilnego rozpoznania → nabijamy fail_count
            self.detect_fail_count += 1
            if self.detect_fail_count >= CONFIG["detect_fail_limit"]:
                # po kilku nieudanych próbach idziemy w PIN_ENTRY
                self.enter_pin_entry()
                return

            # feedback dla usera w overlay
            if conf <= CONFIG["recognition_conf_low"]:
                self.set_message(now_str(), "Nie rozpoznaję…", color="white")
            else:
                self.set_message(now_str(), f"pewność: {conf:.0f}%", color="white")
            return

        # 3) DETECT_RETRY -> po PIN + re-treningu sprawdzamy,
        # czy teraz kamera łapie tego konkretnego pracownika.
        if self.state == "DETECT_RETRY":
            self.detect_retry_count += 1

            if (
                emp_id == self.current_emp_id and
                conf >= CONFIG["recognition_conf_ok"]
            ):
                # Sukces rozpoznania poprawnego pracownika po PIN/treningu
                self.fallback_pin_flag = False

                # uczymy się świeżej próbki
                self._online_learn_face(emp_id)

                # przechodzimy dalej
                self.enter_identified_wait()
                return

            if self.detect_retry_count >= CONFIG["detect_retry_limit"]:
                # Nadal lipa -> fallback_pin_flag True i lecimy dalej
                self.fallback_pin_flag = True
                self.enter_identified_wait()
                return

            # wciąż próbujemy, pokazuj info użytkownikowi
            txt_conf = f"{conf:.0f}%" if conf is not None else ""
            self.set_message(
                "Sprawdzam twarz…",
                f"{self.current_emp_name or ''} {txt_conf}",
                color="white",
            )
            return

        # 4) CALIBRATE -> tylko zbieramy info czy ktoś był/jest blisko
        if self.state == "CALIBRATE":
            if self.last_face_bbox is not None:
                self.calibrate_seen_face = True  # ktoś był widoczny w tym oknie
                (_, _, w, h) = self.last_face_bbox
                if max(w, h) >= CONFIG["face_min_size"]:
                    # Twarz duża => fizycznie blisko ustnika
                    self.calibrate_good_face = True
            return

        # 5) inne stany (IDENTIFIED_WAIT, MEASURE, DECIDE...) -> nie robimy nic
        return


    #################################
    # --- UI tick (zegarek) ---
    #################################
    def on_ui_tick(self):
        if self.state in ("IDLE", "DETECT"):
            # aktualizujemy tylko górny label (czas), nie tykamy koloru
            self.lbl_top.setText(now_str())
            self.lbl_top.setStyleSheet(
                "color:white; font-size:28px; font-weight:600;"
            )
        # inne stany nie aktualizują zegara żeby nie migało komunikatu


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

    @QtCore.pyqtSlot()
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

        # GPIO cleanup
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
    Automatycznie ustawiamy środowisko dla renderowania na lokalnym DSI.
    Dzięki temu można odpalać nawet przez SSH bez ręcznego exportowania.
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

    # obsługa Ctrl+C w konsoli
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
