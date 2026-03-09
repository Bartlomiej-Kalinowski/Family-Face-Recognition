import os
import sys

# Wymuszamy załadowanie silnika YOLO (torch) ZANIM PyQt5 przejmie kontrolę
try:
    from ultralytics import YOLO
    import torch

    _ = torch.empty(1)
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
import numpy as np
from tqdm import tqdm
from config import Config
from database import FaceDatabase
from ml_engine import FaceExtractor, FaceClassifier
from interface import FaceInterface


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
        self.classifier = FaceClassifier()  # Nowa, rozdzielona klasa ML
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
                # Wywołujemy nową metodę z interface.py
                if hasattr(self, 'ui') and self.ui:
                    self.ui.update_progress(i + 1, total, f"Skanowanie: {os.path.basename(full_path)}")

                QApplication.processEvents()

                detected_faces = self.extractor.extract_face_data(full_path)

                for j, face in enumerate(detected_faces):
                    fid = f"{safe_name}_f{j}"
                    clean_emb = np.array(face['embedding']).flatten().astype(np.float32).tolist()

                    # Tu zapisujemy (bez commitu w środku save_face!)
                    self.db.save_face(face['crop'], fid, full_path, face['bbox'], clean_emb)

                self.db.mark_as_processed(full_path)

                # OPTYMALIZACJA: Commitujemy raz po każdym pełnym zdjęciu (nie po każdej twarzy)
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

    def run_classification_phase(self) -> None:
        """Faza 2 ML: Automatyczna predykcja reszty zbioru na podstawie Ground Truth."""
        # 1. Pobierz Ground Truth (to, co człowiek przed chwilą potwierdził)
        labeled_data = self.db.get_labeled_data_for_train()
        if not labeled_data: return

        train_fids, train_labels, train_embs = zip(*labeled_data)

        # 2. Trenuj jeden model dla WSZYSTKICH osób
        self.classifier.train_multiclass_svm(train_embs, train_labels)

        # 3. Pobierz to, co zostało do zgadnięcia
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data: return

        test_fids, test_embs = zip(*unlabeled_data)

        # 4. Wykonaj predykcję na reszcie zbioru
        predictions, probabilities = self.classifier.predict_unlabeled(test_embs)

        matches_count = 0
        for fid, pred_name, prob in zip(test_fids, predictions, probabilities):
            if prob > 0.60:  # Minimalny próg pewności SVM
                self.db.set_svm_prediction(fid, pred_name)
                matches_count += 1

        print(f"SVM sklasyfikował {matches_count} twarzy. Oczekują na weryfikację w GUI.")

    def start(self) -> None:
        """Główna pętla sterująca."""
        mode = self.ui.ask_for_scan_mode()
        if mode != "cancel":
            self.run_initial_scan(mode=mode, limit=500)
            self.run_clustering_phase()
            self.refresh_main_view()
            sys.exit(self.ui.app.exec_())


if __name__ == "__main__":
    app = SmartLabelerController()
    app.start()