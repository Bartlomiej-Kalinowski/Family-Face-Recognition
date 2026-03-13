import json
import os
import shutil
import sys

# Wymuszamy załadowanie silnika YOLO (torch) ZANIM PyQt5 przejmie kontrolę
try:
    from ultralytics import YOLO
    import torch

    _ = torch.empty(1)
except Exception:
    print("Blad importu biblioteki ultralytics lub pytorch")
    exit(0)

from sklearn.metrics import accuracy_score, f1_score, classification_report
from PyQt5.QtWidgets import QApplication
import numpy as np
from tqdm import tqdm
from config import Config
from database import FaceDatabase
from ml_engine import FaceExtractor, FaceClassifier
from interface import FaceInterface
import cv2
from datetime import datetime


class SmartLabelerController:
    """
    Główny kontroler aplikacji (Orkiestrator).
    Wzorzec Architektoniczny: Facade / Controller.
    Łączy warstwę logiki biznesowej (ML, Baza) z interfejsem użytkownika (PyQt).
    """

    def __init__(self):
        self.config = Config()
        self.db = FaceDatabase(self.config)
        self.extractor = FaceExtractor(self.config)
        self.classifier = FaceClassifier()
        self.ui = FaceInterface()


    def _manual_fix_callback(self, face_id: str, new_name: str) -> None:
        """Zapisuje poprawkę wprowadzoną ręcznie w GUI przez użytkownika."""
        print(f"Poprawka ręczna: {face_id} -> {new_name}")
        self.db.set_manual_label(face_id, new_name)
        self.refresh_main_view()

    def refresh_main_view(self) -> None:
        """Pobiera zaktualizowane dane z DB i odświeża siatkę w Frontendzie."""
        labeled_faces = self.db.get_all_labeled_faces()
        self.ui.refresh_classified_faces(labeled_faces, self._manual_fix_callback)

    def run_initial_scan(self, mode: str = "incremental", limit: int = 1000, callback=None) -> None:
        """Ekstrakcja twarzy ze zdjęć (YOLO) i zapis surowych danych do bazy."""
        if mode == "full":
            self.db.clear_database()
            # Opcjonalnie: czyścimy folder z wyciętymi twarzami
            if os.path.exists(self.config.FACES_DIR):
                for f in os.listdir(self.config.FACES_DIR):
                    os.remove(os.path.join(self.config.FACES_DIR, f))

        # Pobieranie ścieżek z normalizacją (rozwiązuje błędy Windows / vs \)
        all_paths = [os.path.normpath(os.path.abspath(os.path.join(r, f)))
                     for r, d, files in os.walk(self.config.SOURCE_DIR)
                     for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Używamy zbioru (set) dla szybszego porównywania
        processed_paths = set(self.db.get_all_processed_paths())
        to_process = [p for p in all_paths if p not in processed_paths][:limit]

        if not to_process:
            print("Brak nowych zdjęć do ekstrakcji.")
            return

        pbar = tqdm(to_process, desc="Ekstrakcja YOLO")
        total = len(to_process)

        for i, full_path in enumerate(pbar):
            safe_name = os.path.basename(full_path).replace(" ", "_").replace(".", "_")
            try:
                # Wywołujemy metodę z interface.py
                if hasattr(self, 'ui') and self.ui:
                    self.ui.update_progress(i + 1, total, f"Skanowanie: {os.path.basename(full_path)}")

                QApplication.processEvents()

                detected_faces = self.extractor.extract_face_data(full_path)

                for j, face in enumerate(detected_faces):
                    fid = f"{safe_name}_f{j}"
                    clean_emb = np.array(face['embedding']).flatten().astype(np.float32).tolist()

                    self.db.save_face(face['crop'], fid, full_path, face['bbox'], clean_emb)

                self.db.mark_as_processed(full_path)

                self.db._conn.commit()

            except Exception as e:
                pbar.write(f"Błąd: {e}")


        total_faces = self.db.get_total_faces_count()
        if hasattr(self, 'ui'):
            self.ui.update_face_stats(total_faces)

        print(f"Skanowanie zakończone. Wycięto i zapisano {total_faces} twarzy.")

    def process_bulk_selection(self, face_ids: list) -> None:
        selected_fids, name = self.ui.bulk_verify_faces(face_ids)
        if not selected_fids or not name: return

        total = len(selected_fids)
        for i, fid in enumerate(selected_fids):
            self.db.set_manual_label(fid, name, is_test=0)

            # Używamy nowej metody z interface.py
            self.ui.update_progress(i + 1, total, f"Zapisywanie: {name}")
            QApplication.processEvents()

        self.db._conn.commit()  # SZYBKI ZAPIS NA KOŃCU

    def run_clustering_phase(self) -> None:
        """Faza 1 ML: Pół-nadzorowane etykietowanie z DBSCAN."""
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data: return

        # print("Ilosc twarzy procesowanych przez DBSCAN:\t", len(unlabeled_data))

        fids, embeddings = [item[0] for item in unlabeled_data], np.array([item[1] for item in unlabeled_data])

        # DBSCAN delegowany do osobnego modułu ML
        clusters = self.classifier.get_face_clusters(embeddings, fids)
        valid_clusters = {cid: cfids for cid, cfids in clusters.items() if len(cfids) >= 3}

        print("Valid clusters number:\t", len(clusters))
        print("Valid clusters number:\t", len(valid_clusters))

        for cluster_id, cluster_fids in valid_clusters.items():
            self.process_bulk_selection(cluster_fids[:13])  # Limit do sprawdzenia

        labeled_count = len(set(label for _, label, _ in self.db.get_labeled_data_for_train()))

        # Po ręcznym zaetykietowaniu klastrów, odpalamy wieloklasowy SVM na reszcie
        if labeled_count >= 2:
            print(f"Mamy {labeled_count} osoby. Uruchamiam SVM...")
            self.run_classification_phase()
        else:
            print("Zbyt mało osób w bazie (wymagane min. 2), aby uruchomić SVM.")


    def run_classification_phase(self):
        """Faza 1: Przygotowanie danych i trenowanie modelu SVM."""
        print("\n[SYSTEM] Rozpoczynanie fazy treningu...")

        # Przenosimy niepodpisane do zbioru testowego
        self.db.mark_unlabeled_as_test()

        # 1. Pobieranie danych treningowych
        train_data = self.db.get_labeled_data_for_train()
        if not train_data:
            print("[BŁĄD] Brak danych treningowych w bazie.")
            return

        unique_labels = set(label for _, label, _ in train_data)
        if len(unique_labels) < 2:
            print(f"[BŁĄD] Zbyt mało osób ({len(unique_labels)}). Potrzeba min. 2 do SVM.")
            return

        # Rozpakowanie (fid, label, embedding)
        _, train_labels, train_embs = zip(*train_data)

        # 2. Trening modelu
        self.classifier.train_multiclass_svm(list(train_embs), list(train_labels))

        # Przejdź do fazy testowania i raportowania
        self.run_evaluation_phase(train_data)

    def run_evaluation_phase(self, train_data):
        """Faza 2: Predykcja na zbiorze testowym, generowanie raportu i logowanie."""

        # 1. Pobieranie danych testowych (z nowej metody w database.py)
        test_data = self.db.get_unlabeled_test_data()

        if not test_data:
            print("[INFO] Brak nowych danych testowych do klasyfikacji.")
            return

        # Rozpakowanie danych (fids, paths, test_embs_np)
        # UWAGA: test_embs są już obiektami numpy dzięki poprawce w database.py
        fids, paths, test_embs, bbox = zip(*test_data)

        # 2. Predykcja
        y_pred, confidences = self.classifier.predict_unlabeled(list(test_embs))
        if len(y_pred) == 0:
            print("[BŁĄD] Model nie zwrócił predykcji.")
            return

        # Ground Truth z nazw plików
        y_true = [self.get_gt_from_path(p) for p in paths]

        # 3. Wyliczanie metryk
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)

        # 4. Logowanie do pliku
        log_path = "wyniki_klasyfikacji.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"SESJA: {timestamp}\n")
            f.write(f"Próbki: Trening={len(train_data)}, Test={len(test_data)}\n")
            f.write(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Raport szczegółowy:\n")
            f.write(report)
            f.write(f"{'=' * 60}\n")

        print(f"\n[SUKCES] Raport zapisany w: {log_path}")
        print(report)

        # 5. Zapis wyników do bazy danych
        for fid, pred in zip(fids, y_pred):
            self.db.set_svm_prediction(fid, pred)

        # 6. Aktualizacja GUI
        classified_list = list(zip(fids, y_pred))
        self.ui.refresh_classified_faces(classified_list, self._manual_fix_callback)

        # 7. POPUP: Pytanie o wizualizację

        reply = self.ui.confirm_all_labels()

        if reply == 0:
            self.draw_all_labels_on_faces(self.config.ANNOTATED_FACES_DIR)


    def start(self) -> None:
        """Główna pętla sterująca z obsługą istniejącego zbioru."""
        mode = self.ui.ask_for_scan_mode()

        if mode == "use_existing":
            # OPCJA NOWA: Buduj bazę z folderu
            self.rebuild_db_from_files()
            self.run_clustering_phase()  # Od razu przejdź do klastrowania/SVM

        elif mode in ["full", "incremental"]:
            # OPCJA STARA: Skanuj oryginalne zdjęcia przez YOLO
            self.run_initial_scan(mode=mode, limit=500)
            self.run_clustering_phase()

        elif mode == "cancel":
            return

        self.refresh_main_view()
        sys.exit(self.ui.app.exec_())

    def get_gt_from_path(self, path):
        # Wyciąga 'Jan_Kowalski' z 'C:/sciezka/Jan_Kowalski_01.jpg'
        filename = os.path.basename(path)
        name_part = os.path.splitext(filename)[0]  # Usuwa .jpg
        parts = name_part.split('_')
        # Jeśli na końcu jest numer (np. _01), odcinamy go
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return name_part

    def rebuild_db_from_files(self):
        """Czyści bazę i buduje ją od nowa na podstawie plików w extracted_faces."""
        print("Rozpoczynam odbudowę bazy na podstawie istniejących wycinków...")

        # 1. Czyścimy tylko bazę danych (pliki zostają!)
        self.db.clear_database()

        face_folder = self.config.FACES_DIR
        files = [f for f in os.listdir(face_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not files:
            print("Folder z twarzami jest pusty!")
            return

        total = len(files)
        for i, filename in enumerate(files):
            full_path = os.path.join(face_folder, filename)
            face_img = cv2.imread(full_path)

            if face_img is None:
                continue

            # Wyciągamy etykietę (Ground Truth) z nazwy pliku
            # Przyjmujemy, że plik ma nazwę Jan_Kowalski_01.jpg
            gt_label = self.get_gt_from_path(full_path)

            # Generujemy embedding (używając Twojego modelu FaceExtractor)
            # Musimy przygotować obraz tak, jak robi to YOLO
            gray = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            embedding = np.array(hog.compute(gray)).flatten().astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            # Zapisujemy do bazy
            # face_id to nazwa pliku bez rozszerzenia
            face_id = os.path.splitext(filename)[0]

            # Tutaj używamy bezpośrednio kursora, żeby zapisać też Ground Truth jeśli chcesz,
            # albo po prostu zapisujemy jako normalną twarz:
            emb_json = json.dumps(embedding.tolist())
            self.db._cursor.execute('''
                INSERT INTO faces (face_id, image_path, embedding, is_test)
                VALUES (?, ?, ?, ?)
            ''', (face_id, full_path, emb_json, 0))

            # Aktualizujemy pasek postępu w GUI jeśli dostępny
            if hasattr(self, 'ui'):
                self.ui.update_progress(i + 1, total, f"Import: {gt_label}")
                QApplication.processEvents()

        self.db._conn.commit()
        print(f"Baza odbudowana. Zaimportowano {total} twarzy.")

    def draw_all_labels_on_faces(self, target_dir):
        """
        Nanosi etykiety na kopie wycinków twarzy.
        Odróżnia twarze podpisane ręcznie od tych przewidzianych przez model.
        """

        print("[SYSTEM] Generowanie boksów i etykiet na wszystkich twarzach...")

        # Wyciągamy WSZYSTKIE podpisane twarze z bazy.`
        # COALESCE wybiera pierwsze niepuste pole (najpierw manualne, potem SVM).
        # Zwracamy też 'manual_label', żeby wiedzieć, czy etykieta pochodzi od człowieka.
        results = self.db.get_all_labeled_faces()

        for face_id, label, is_manual, original_img_path in results:
            img_path = os.path.join(target_dir, f"{face_id}.jpg")

            if not os.path.exists(img_path):
                shutil.copy2(original_img_path, img_path)

            img = cv2.imread(img_path)
            if img is None: continue

            h, w, _ = img.shape

            # --- PARAMETRY WIZUALNE ---
            # Jeśli etykieta była nadana ręcznie -> ZIELONA (100% pewności)
            # Jeśli etykieta pochodzi z predykcji SVM -> POMARAŃCZOWA (Przewidywanie)
            color = (0, 255, 0) if is_manual else (0, 165, 255)
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, w / 300)

            # Rysowanie ramki na wycinku
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)

            # Etykieta tekstu (z dodanym tagiem źródła)
            source_tag = "[MANUAL]" if is_manual else "[SVM]"
            label_text = f"{source_tag} {label.upper()}"

            # Tło i tekst
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)
            cv2.rectangle(img, (0, 0), (lbl_w + 10, lbl_h + 15), color, -1)
            cv2.putText(img, label_text, (5, lbl_h + 8), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # Zapis pliku
            cv2.imwrite(img_path, img)

        # Informacja zwrotna dla użytkownika w interfejsie
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(None, "Sukces", f"Zapisano wizualizacje w:\n{target_dir}")
        print(f"[SUKCES] Folder '{target_dir}' został zaktualizowany o ramki.")




if __name__ == "__main__":
    app = SmartLabelerController()
    app.start()