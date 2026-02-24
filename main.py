import os
import sys

# 1. Rozwiązanie problemu z c10.dll/OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Wymuszamy załadowanie silnika YOLO (torch) ZANIM PyQt5 przejmie kontrolę
try:
    from ultralytics import YOLO
    import torch
    _ = torch.empty(1)
except Exception:
    pass

import numpy as np
from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from config import Config
from database import FaceDatabase
from model_engine import FaceEngine
from interface import FaceInterface
from sklearn.decomposition import PCA


class SmartLabeler:
    def __init__(self):
        self.config = Config()
        self.db = FaceDatabase(self.config)
        self.engine = FaceEngine(self.config)
        self.ui = FaceInterface()

    def manual_fix_callback(self, face_id, new_name):
        """
        Wywoływane z GUI (FaceCard), gdy użytkownik zmieni imię
        w głównym oknie weryfikacji.
        """
        print(f"Ręczna korekta: {face_id} -> {new_name}")
        # Ustawiamy validated=True, bo użytkownik osobiście to potwierdził
        self.db.set_label(face_id, new_name, validated=True)

        # Opcjonalnie: możemy tutaj ponownie odpalić SVM dla tego imienia,
        # aby douczyć model na podstawie poprawki.

    def refresh_main_view(self):
        """Pobiera dane z bazy i odświeża siatkę w głównym oknie."""
        labeled_faces = self.db.get_all_labeled_faces()
        # Przekazujemy listę krotek i metodę callback do obsługi zmian
        self.ui.refresh_classified_faces(labeled_faces, self.manual_fix_callback)

    def run_initial_scan(self, limit=600):
        print("\n--- Przygotowanie do nowego skanu (czyszczenie bazy) ---")
        self.db.clear_database()

        # Opcjonalnie: usuwanie starych wyciętych twarzy z folderu
        if os.path.exists(self.config.FACES_DIR):
            for file in os.listdir(self.config.FACES_DIR):
                file_path = os.path.join(self.config.FACES_DIR, file)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                except Exception as e:
                    print(f"Nie udało się usunąć {file}: {e}")
        else:
            os.makedirs(self.config.FACES_DIR, exist_ok=True)
        # 1. Zbieranie ścieżek
        image_paths = []
        for root, _, files in os.walk(self.config.SOURCE_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                if len(image_paths) >= limit: break
            if len(image_paths) >= limit: break

        processed_paths = self.db.get_all_processed_paths()

        # 2. Inicjalizacja paska w GUI
        total_to_scan = len(image_paths)
        self.ui.show_startup_progress(total_to_scan)

        # 3. Pętla skanowania
        for i, full_path in enumerate(tqdm(image_paths, desc="Feature extraction", unit="img")):

            # Aktualizacja GUI
            self.ui.update_startup_progress(i, f"Przetwarzanie: {os.path.basename(full_path)}")

            if full_path in processed_paths:
                continue

            detected_faces = self.engine.extract_face_data(full_path)
            file_name = os.path.basename(full_path)

            for j, face_data in enumerate(detected_faces):
                face_id = f"{os.path.splitext(file_name)[0]}_f{j}"
                self.db.save_face(face_data['crop'], face_id, full_path, face_data['bbox'])
                self.db.update_embedding(face_id, face_data['embedding'])

        # 4. Zamknięcie paska przed startem właściwego etykietowania
        self.ui.close_startup_progress()
        print(f"\nScan complete.")

    def process_bulk_selection(self, face_ids):
        selected_fids, name = self.ui.bulk_verify_faces(face_ids)
        if not selected_fids or not name:
            return

        embeddings_for_svm = []
        for fid in selected_fids:
            self.db.set_label(fid, name, validated=True)
            meta = self.db.get_metadata_for_gui(fid)
            embeddings_for_svm.append(meta['embedding'])

        # SVM podpisuje resztę w tle
        self._expand_identity_from_group(name, embeddings_for_svm)
        # Odśwież widok po działaniu SVM
        self.refresh_main_view()

    def _expand_identity_from_group(self, name, group_embeddings):
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data: return

        fids = [item[0] for item in unlabeled_data]
        test_embs = np.array([item[1] for item in unlabeled_data])

        # Trenujemy One-Class SVM na grupie (to jest dużo stabilniejsze!)
        train_embs = np.array(group_embeddings)

        # Ustawiamy nu (margines błędu) - im mniejszy, tym ostrzejszy model
        from sklearn.svm import OneClassSVM
        clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        clf.fit(train_embs)

        # Decyzja: score_samples podaje dystans od nauczonej grupy
        # Im wyższa wartość, tym bardziej twarz pasuje do grupy
        scores = clf.decision_function(test_embs)

        matches_found = 0
        for fid, score in zip(fids, scores):
            if score > 0:  # 0 to granica wyznaczona przez nu
                self.db.set_label(fid, name, validated=False)
                matches_found += 1

        print(f"SVM przeanalizował grupę i podpisał dodatkowe {matches_found} zdjęć.")

    def start_labeling(self):
        """Finalizuje proces i utrzymuje okno otwarte dla użytkownika."""
        print("\n--- Wszystkie fazy zakończone. Panel weryfikacji gotowy. ---")
        self.refresh_main_view()

        # Uruchomienie pętli zdarzeń Qt (zapobiega zamknięciu okna)
        sys.exit(self.ui.app.exec_())

    def run_clustering_phase(self):
        print("\n--- Rozpoczynam fazę grupowania (DBSCAN) ---")
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data:
            print("Brak niepodpisanych danych.")
            return

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        # PCA dla stabilności
        pca = PCA(n_components=min(50, len(embeddings)))
        reduced_embeddings = pca.fit_transform(embeddings)

        clusters = self.engine.get_face_clusters(reduced_embeddings, fids)
        print(f"Znaleziono {len(clusters)} grup.")

        for cluster_id, cluster_fids in clusters.items():
            # Wywołanie okna modalnego
            self.process_bulk_selection(cluster_fids[:16])
            # Odświeżanie po każdej grupie
            self.refresh_main_view()


if __name__ == "__main__":
    app = SmartLabeler()
    app.run_initial_scan(limit=600)
    app.run_clustering_phase()
    app.start_labeling()