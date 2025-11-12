# facedb.py
import os
import glob
import json
import cv2
import numpy as np
from datetime import datetime

from config import CONFIG

# Ścieżki modeli (zgodnie z Twoim układem: /models)
_YUNET_PATH = CONFIG.get("yunet_model_path", "models/face_detection_yunet_2023mar.onnx")
_SSD_PROTO  = CONFIG.get("ssd_prototxt_path", CONFIG.get("ssd_proto_path", "models/deploy.prototxt"))
_SSD_MODEL  = CONFIG.get("ssd_model_path",     CONFIG.get("ssd_caffemodel_path", "models/res10_300x300_ssd_iter_140000.caffemodel"))

class FaceDB:
    """
    Struktura danych / plików:

    employees.json:
      {
        "employees": [
          {"id":"1","name":"Kamil Karolak","pin":"0000"},
          ...
        ]
      }

    faces/<id>/*.jpg  – przycięte 240x240 BGR

    index/<id>.npz    – lista deskryptorów ORB zapisanych po train_reindex()

    W RAM:
      self.index[emp_id] = [desc_arr1, desc_arr2, ...]
    """

    def __init__(self, faces_dir, index_dir, employees_json):
        self.faces_dir = faces_dir
        self.index_dir = index_dir
        self.employees_json = employees_json

        self._load_employees()

        # Detektory twarzy: YuNet → SSD → Haar (fallback)
        self._init_detectors()
        self.cascade = cv2.CascadeClassifier(self._find_haar())

        # Ekstraktor cech
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Indeks deskryptorów
        self.index = {}
        self._load_index()

    # ---------- Detektory twarzy ----------
    def _init_detectors(self):
        """Przygotuj detektory: YuNet (FaceDetectorYN) -> SSD (DNN) -> Haar (fallback)."""
        self._det_yunet = None
        self._det_ssd = None

        # YuNet (OpenCV Zoo, API FaceDetectorYN)
        try:
            if hasattr(cv2, "FaceDetectorYN_create") and os.path.exists(_YUNET_PATH):
                score_th = float(CONFIG.get("yunet_score_thresh", 0.85))
                nms_th   = float(CONFIG.get("yunet_nms_thresh",   0.3))
                top_k    = int(CONFIG.get("yunet_top_k", 5000))
                # input size ustawiamy dynamicznie w _detect_faces
                self._det_yunet = cv2.FaceDetectorYN_create(
                    _YUNET_PATH, "", (320, 320), score_th, nms_th, top_k
                )
        except Exception:
            self._det_yunet = None

        # SSD ResNet-10 (DNN)
        try:
            if os.path.exists(_SSD_PROTO) and os.path.exists(_SSD_MODEL):
                self._det_ssd = cv2.dnn.readNetFromCaffe(_SSD_PROTO, _SSD_MODEL)
        except Exception:
            self._det_ssd = None

    def _detect_faces(self, img_bgr):
        """
        Zwraca listę [(x,y,w,h), ...] w pikselach.
        Priorytet: YuNet -> SSD -> Haar.
        """
        h, w = img_bgr.shape[:2]

        # YuNet
        if self._det_yunet is not None:
            try:
                self._det_yunet.setInputSize((w, h))
                retval, faces = self._det_yunet.detect(img_bgr)
                boxes = []
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        x, y, ww, hh = f[:4]
                        boxes.append((int(x), int(y), int(ww), int(hh)))
                if boxes:
                    return boxes
            except Exception:
                pass

        # SSD (DNN)
        if self._det_ssd is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    img_bgr, 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False
                )
                self._det_ssd.setInput(blob)
                det = self._det_ssd.forward()
                conf_th = float(CONFIG.get("ssd_conf_thresh", 0.70))
                boxes = []
                # det: [1,1,N,7] => (batch, class, i, [id, conf, x1,y1,x2,y2])
                for i in range(det.shape[2]):
                    conf = float(det[0, 0, i, 2])
                    if conf >= conf_th:
                        x1 = int(det[0, 0, i, 3] * w)
                        y1 = int(det[0, 0, i, 4] * h)
                        x2 = int(det[0, 0, i, 5] * w)
                        y2 = int(det[0, 0, i, 6] * h)
                        xx1, yy1 = max(0, x1), max(0, y1)
                        xx2, yy2 = min(w - 1, x2), min(h - 1, y2)
                        if xx2 > xx1 and yy2 > yy1:
                            boxes.append((xx1, yy1, xx2 - xx1, yy2 - yy1))
                if boxes:
                    return boxes
            except Exception:
                pass

        # Haar (fallback)
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.2, 5)
            return [(int(x), int(y), int(ww), int(hh)) for (x, y, ww, hh) in faces]
        except Exception:
            return []

    # ---------- Init helpers ----------
    def _find_haar(self):
        """Znajdź haarcascade_frontalface_default.xml w typowych ścieżkach."""
        candidates = []
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidates.append(cv2.data.haarcascades)
        candidates += [
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
        ]
        fname = "haarcascade_frontalface_default.xml"
        for base in candidates:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        return fname  # fallback: bieżący katalog

    def _load_employees(self):
        """employees.json -> self.employees, self.emp_by_pin, self.emp_by_id"""
        with open(self.employees_json, "r", encoding="utf-8") as f:
            self.employees = json.load(f)

        self.emp_by_pin = {
            e["pin"]: e
            for e in self.employees.get("employees", [])
            if "pin" in e
        }
        self.emp_by_id = {
            (e.get("id") or e.get("name")): e
            for e in self.employees.get("employees", [])
        }

    def save_employees(self):
        with open(self.employees_json, "w", encoding="utf-8") as f:
            json.dump(self.employees, f, ensure_ascii=False, indent=2)
        self._load_employees()

    def ensure_employee_exists(self, emp_id: str, name: str, pin: str):
        found = any((e.get("id") == emp_id) for e in self.employees["employees"])
        if not found:
            self.employees["employees"].append({"id": emp_id, "name": name, "pin": pin})
            self.save_employees()
        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    # ---------- Zrzuty twarzy ----------
    def add_three_shots(self, emp_id: str, imgs_bgr_list):
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        for img in imgs_bgr_list:
            outp = os.path.join(
                folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            )
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def _enforce_faces_limit(self, emp_id: str, max_len: int):
        folder = os.path.join(self.faces_dir, emp_id)
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        overflow = len(files) - max_len
        if overflow > 0:
            for p in files[0:overflow]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def add_online_face_sample(self, emp_id: str, face_bgr_240):
        gray = cv2.cvtColor(face_bgr_240, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return False
        if emp_id not in self.index:
            self.index[emp_id] = []
        self.index[emp_id].append(desc)
        max_len = CONFIG.get("online_max_samples_per_emp", 20)
        if len(self.index[emp_id]) > max_len:
            self.index[emp_id] = self.index[emp_id][-max_len:]

        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        outp = os.path.join(folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
        cv2.imwrite(outp, face_bgr_240, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self._enforce_faces_limit(emp_id, max_len)

        self._save_index_for(emp_id, self.index[emp_id])
        return True

    # ---------- Index ORB ----------
    def _load_index(self):
        self.index = {}
        for e in self.employees.get("employees", []):
            emp_id = e.get("id") or e.get("name")
            npz_path = os.path.join(self.index_dir, f"{emp_id}.npz")
            if os.path.exists(npz_path):
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    self.index[emp_id] = list(npz.get("descriptors", []))
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
                progress_callback(i + 1, n)

    # ---------- Rozpoznawanie ----------
    def recognize_face(self, img_bgr):
        """
        Zwraca:
            (emp_id or None,
             display_name or None,
             confidence%,
             bbox=(x,y,w,h) or None)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = self._detect_faces(img_bgr)
        if not faces:
            return None, None, 0.0, None

        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (240, 240), interpolation=cv2.INTER_LINEAR)

        _, desc = self.orb.detectAndCompute(roi_gray, None)
        if desc is None or len(desc) == 0:
            return None, None, 0.0, (x, y, w, h)

        ratio_th  = CONFIG["recognition_ratio_thresh"]
        min_match = CONFIG["recognition_min_match"]
        min_margin= CONFIG["recognition_min_margin"]

        knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        best_emp = None
        best_score = 0
        second_best = 0

        for emp_id, desc_list in self.index.items():
            emp_score = 0
            for dset in desc_list:
                if dset is None or len(dset) == 0:
                    continue
                matches_knn = knn_matcher.knnMatch(desc, dset, k=2)
                for m in matches_knn:
                    if len(m) < 2:
                        continue
                    m1, m2 = m[0], m[1]
                    if m1.distance < ratio_th * m2.distance:
                        emp_score += 1

            if emp_score > best_score:
                second_best, best_score, best_emp = best_score, emp_score, emp_id
            elif emp_score > second_best:
                second_best = emp_score

        if best_score < min_match:
            return None, None, 0.0, (x, y, w, h)
        if (best_score - second_best) < min_margin:
            return None, None, 0.0, (x, y, w, h)

        total = max(1, best_score + second_best)
        conf = min(100.0, 100.0 * (best_score / total))

        display_name = None
        if best_emp:
            emp_entry = self.emp_by_id.get(best_emp)
            display_name = emp_entry.get("name", best_emp) if emp_entry else best_emp

        return best_emp, display_name, conf, (x, y, w, h)
