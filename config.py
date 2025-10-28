# config.py
# Konfiguracja globalna aplikacji (przeniesiona 1:1 z monolitu)

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

    # --- rozpoznawanie twarzy
    "face_detect_interval_ms": 1000,  # jak często detekcja/ID
    "face_min_size": 120,
    "recognition_conf_ok": 55.0,   # %
    "recognition_conf_low": 20.0,  # %

    # ile prób w DETECT zanim wymusimy PIN
    "detect_fail_limit": 5,
    # ile prób w DETECT_RETRY zanim uznamy fallback_pin_flag
    "detect_retry_limit": 3,

    # --- jakość próbek do treningu po PIN ---
    "train_required_shots": 7,
    "train_timeout_sec": 15,

    # progi jakości pojedynczego ujęcia
    "quality_min_sharpness": 60.0,
    "quality_min_brightness": 40.0,
    "quality_max_brightness": 210.0,

    # --- rozpoznawanie (anty-false-positive) ---
    "recognition_min_match": 25,
    "recognition_ratio_thresh": 0.75,
    "recognition_min_margin": 10,
    "recognition_stable_ticks": 2,

    # --- on-line uczenie twarzy ---
    "online_max_samples_per_emp": 20,


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
