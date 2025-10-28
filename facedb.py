# facedb.py
import os
import glob
import json
import cv2
import numpy as np
from datetime import datetime

from config import CONFIG


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

    faces/<id>/*.jpg  – zdjecia twarzy konkretnego pracownika,
                         docelowo przycięte i przeskalowane do ~240x240 BGR

    index/<id>.npz    – ORB deskryptory twarzy (lista tablic ORB descriptors)
                        zapisane po train_reindex()

    W RAM:
      self.index[emp_id] = [desc_arr1, desc_arr2, ...]
    """

    def __init__(self, faces_dir, index_dir, employees_json):
        self.faces_dir = faces_dir
        self.index_dir = index_dir
        self.employees_json = employees_json

        self._load_employees()

        # Detektor twarzy
        self.cascade = cv2.CascadeClassifier(self._find_haar())

        # Ekstraktor cech
        self.orb = cv2.ORB_create(nfeatures=1000)

        # BFMatcher bezpośrednio w recognize_face robimy KNN (crossCheck=False)
        self.index = {}
        self._load_index()

    # -------------------------
    # Init helpers
    # -------------------------
    def _find_haar(self):
        """
        Znajdź haarcascade_frontalface_default.xml.
        Szukamy w standardowych ścieżkach OpenCV na RPi.
        """
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

        # fallback – może być w cwd
        return fname

    def _load_employees(self):
        """
        employees.json -> self.employees, self.emp_by_pin, self.emp_by_id
        """
        with open(self.employees_json, "r", encoding="utf-8") as f:
            self.employees = json.load(f)

        # lookup po PIN
        self.emp_by_pin = {
            e["pin"]: e
            for e in self.employees.get("employees", [])
            if "pin" in e
        }

        # lookup po ID
        self.emp_by_id = {
            (e.get("id") or e.get("name")): e
            for e in self.employees.get("employees", [])
        }

    def save_employees(self):
        """
        Zapisuje self.employees do employees.json i odświeża lookupy.
        """
        with open(self.employees_json, "w", encoding="utf-8") as f:
            json.dump(self.employees, f, ensure_ascii=False, indent=2)
        self._load_employees()

    def ensure_employee_exists(self, emp_id: str, name: str, pin: str):
        """
        Upewnia się, że podany pracownik jest w employees.json.
        Z punktu widzenia wdrożenia produkcyjnego:
        - zakład ma listę pracowników z PINami już wpisaną,
        - my tylko dopilnujemy, żeby katalog faces/<id> istniał.

        ALE: zostawiamy to tak, jak w monolicie,
        bo demo/test może startować z czystą bazą.
        """
        found = False
        for e in self.employees["employees"]:
            if e.get("id") == emp_id:
                found = True
                break
        if not found:
            self.employees["employees"].append(
                {"id": emp_id, "name": name, "pin": pin}
            )
            self.save_employees()

        os.makedirs(os.path.join(self.faces_dir, emp_id), exist_ok=True)

    # -------------------------
    # Zrzuty twarzy
    # -------------------------
    def add_three_shots(self, emp_id: str, imgs_bgr_list):
        """
        Zapisuje ZBIÓR próbek twarzy (BGR, najlepiej 240x240) dla pracownika emp_id.

        Nazwa historyczna "add_three_shots" została, ale teraz może to być 3...N
        obrazów, bo w main.collect_new_shots_for_current_emp() zbieramy
        train_required_shots dobrej jakości.

        Każdy zapisujemy jako JPEG ~90%.
        """
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)

        for img in imgs_bgr_list:
            outp = os.path.join(
                folder,
                datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            )
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def _enforce_faces_limit(self, emp_id: str, max_len: int):
        """
        Utrzymuje max_len najnowszych plików JPG w faces/<emp_id>/.
        Sortujemy po nazwie (a nazwa = timestamp), więc najstarsze będą pierwsze.
        Jeśli mamy za dużo, kasujemy najstarsze.
        """
        folder = os.path.join(self.faces_dir, emp_id)
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        overflow = len(files) - max_len
        if overflow > 0:
            to_del = files[0:overflow]
            for p in to_del:
                try:
                    os.remove(p)
                except Exception:
                    # jeżeli nie uda się skasować, trudno – nie crashujemy całego GUI
                    pass

    def add_online_face_sample(self, emp_id: str, face_bgr_240):
        """
        Dodaj pojedynczą NOWĄ próbkę twarzy (już przyciętą i przeskalowaną do 240x240 BGR)
        dla konkretnego pracownika, w momencie kiedy kamera była PEWNA kto to jest.

        Co robimy:
        - wyciągamy ORB deskryptory z tej próbki,
        - dopisujemy do self.index[emp_id] (RAM),
        - jeżeli jest > online_max_samples_per_emp, przycinamy do ostatnich N,
        - zapisujemy .npz dla tego emp_id,
        - zapisujemy JPG z timestampem do faces/<emp_id>/ i tam również pilnujemy limitu.

        Zwraca True jeśli wszystko poszło sensownie (mamy deskryptory),
        False jeśli z jakiegoś powodu próbką nie warto się uczyć (np. ORB nic nie znalazł).
        """

        # Konwersja do szarości żeby wyciągnąć ORB (dokładnie tak jak w train_reindex)
        gray = cv2.cvtColor(face_bgr_240, cv2.COLOR_BGR2GRAY)

        # policz cechy ORB na tym ujęciu
        _, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return False  # próbka bezużyteczna (np. kompletnie rozmazana)

        # dopisz do RAM (self.index[emp_id] to lista tablic numpy N x 32)
        if emp_id not in self.index:
            self.index[emp_id] = []
        self.index[emp_id].append(desc)

        # ogranicz do ostatnich N próbek
        max_len = CONFIG.get("online_max_samples_per_emp", 20)
        if len(self.index[emp_id]) > max_len:
            self.index[emp_id] = self.index[emp_id][-max_len:]

        # zapisz obraz 240x240 jako jpg ~90%
        folder = os.path.join(self.faces_dir, emp_id)
        os.makedirs(folder, exist_ok=True)
        outp = os.path.join(
            folder,
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        )
        # uwaga: cv2.imwrite zapisuje w BGR -> plik JPEG (to jest normalne)
        # IMWRITE_JPEG_QUALITY=90 daje sensowną jakość bez zajmowania 500KB
        cv2.imwrite(outp, face_bgr_240, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # ogranicz liczbę jpg w katalogu pracownika
        self._enforce_faces_limit(emp_id, max_len)

        # zaktualizuj plik index/<emp_id>.npz tylko dla tego jednego pracownika
        self._save_index_for(emp_id, self.index[emp_id])

        return True


    # -------------------------
    # Index ORB (uczenie / odświeżanie)
    # -------------------------
    def _load_index(self):
        """
        Wczytuje deskryptory ORB z index/<emp_id>.npz do self.index[emp_id].
        """
        self.index = {}
        for e in self.employees.get("employees", []):
            emp_id = e.get("id") or e.get("name")
            npz_path = os.path.join(self.index_dir, f"{emp_id}.npz")

            if os.path.exists(npz_path):
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    if "descriptors" in npz:
                        self.index[emp_id] = list(npz["descriptors"])
                    else:
                        self.index[emp_id] = []
                except Exception:
                    # plik uszkodzony?
                    self.index[emp_id] = []
            else:
                self.index[emp_id] = []

    def _save_index_for(self, emp_id: str, descriptors_list):
        """
        descriptors_list: [desc_arr1, desc_arr2, ...] gdzie każda desc_arr
        to numpy array (N x 32) od ORB.
        """
        os.makedirs(self.index_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.index_dir, f"{emp_id}.npz"),
            descriptors=np.array(descriptors_list, dtype=object)
        )

    def train_reindex(self, progress_callback=None):
        """
        Przelatuje po wszystkich folderach faces/<id>/,
        detekcja twarzy, ORB z ROI, zapis deskryptorów do index/<id>.npz
        i odświeżenie self.index.

        To samo co w monolicie, tylko utrzymane 1:1 logicznie.
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

                # wybieramy największą twarz w kadrze (jeśli jest)
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                    roi = gray[y:y+h, x:x+w]
                else:
                    # jak nie ma twarzy (powinno się nie zdarzać po filtrach jakości,
                    # ale na wszelki wypadek) bierzemy cały grayscale
                    roi = gray

                # normalizacja rozmiaru na stałe wejście dla ORB
                roi = cv2.resize(roi, (240, 240), interpolation=cv2.INTER_LINEAR)

                # wyciągnięcie cech ORB
                _, desc = self.orb.detectAndCompute(roi, None)
                if desc is not None and len(desc) > 0:
                    desc_list.append(desc)

            # zapisz deskryptory dla tego emp_id
            self.index[emp_id] = desc_list
            self._save_index_for(emp_id, desc_list)

            if progress_callback:
                progress_callback(i + 1, n)

    # -------------------------
    # Rozpoznawanie twarzy
    # -------------------------
    def recognize_face(self, img_bgr):
        """
        Wejście:
            img_bgr – aktualna ramka z kamery już po obrocie,
                      dokładnie taka jak user widzi na ekranie.

        Zwraca:
            (emp_id or None,
             display_name or None,
             confidence%,           (dla UI)
             bbox=(x,y,w,h) or None)

        MECHANIKA:
        1. Haar cascade -> wykrywamy twarze.
        2. Bierzemy największą.
        3. ORB na wyciętej twarzy.
        4. Dla każdego pracownika porównujemy desc z jego deskryptorami:
            - BFMatcher.knnMatch(k=2) z ratio testem
        5. Liczymy "emp_score" = ile par przeszło ratio test.
        6. Wybieramy best_emp (najwyższy emp_score) + second_best.
        7. Zabezpieczenia anty-false-positive:
            - best_score >= recognition_min_match
            - (best_score - second_best) >= recognition_min_margin
        8. confidence do UI: jaki procent przewagi best nad sumą best+second.

        Progi bierzemy z CONFIG:
            recognition_ratio_thresh
            recognition_min_match
            recognition_min_margin
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
            return None, None, 0.0, None

        # Największa twarz
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (240, 240), interpolation=cv2.INTER_LINEAR)

        # Cechy ORB tej twarzy
        _, desc = self.orb.detectAndCompute(roi_gray, None)
        if desc is None or len(desc) == 0:
            return None, None, 0.0, (x, y, w, h)

        # Ustawienia progowe z configu
        ratio_th = CONFIG["recognition_ratio_thresh"]
        min_match = CONFIG["recognition_min_match"]
        min_margin = CONFIG["recognition_min_margin"]

        # KNN matcher (ratio test)
        knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        best_emp = None
        best_score = 0
        second_best = 0

        for emp_id, desc_list in self.index.items():
            emp_score = 0

            for dset in desc_list:
                if dset is None or len(dset) == 0:
                    continue

                # dopasowania k=2
                matches_knn = knn_matcher.knnMatch(desc, dset, k=2)

                for m in matches_knn:
                    if len(m) < 2:
                        continue

                    m1, m2 = m[0], m[1]
                    # klasyczny ratio test Lowe'a:
                    if m1.distance < ratio_th * m2.distance:
                        emp_score += 1

            # aktualizuj best/second
            if emp_score > best_score:
                second_best, best_score, best_emp = best_score, emp_score, emp_id
            elif emp_score > second_best:
                second_best = emp_score

        # Twarde odrzucenie, żeby ktoś obcy nie wszedł jako pierwszy z brzegu
        if best_score < min_match:
            # za mało cech pasuje -> nie rozpoznajemy nikogo
            return None, None, 0.0, (x, y, w, h)

        if (best_score - second_best) < min_margin:
            # za mała przewaga nad drugim miejscem -> niepewne
            return None, None, 0.0, (x, y, w, h)

        # "confidence" tylko do pokazania kolorków i tekstu,
        # nie jest już jedynym kryterium.
        total = max(1, best_score + second_best)
        conf = min(100.0, 100.0 * (best_score / total))

        # Nazwa do UI
        display_name = None
        if best_emp:
            emp_entry = self.emp_by_id.get(best_emp)
            if emp_entry:
                display_name = emp_entry.get("name", best_emp)
            else:
                display_name = best_emp

        return best_emp, display_name, conf, (x, y, w, h)
