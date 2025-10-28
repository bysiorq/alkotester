# utils_fs.py
import os
import json
from datetime import datetime
from config import CONFIG

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
