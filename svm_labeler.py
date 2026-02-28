import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Wymuszamy załadowanie silnika YOLO (torch) ZANIM PyQt5 przejmie kontrolę
try:
    from ultralytics import YOLO
    import torch

    _ = torch.empty(1)
except Exception:
    pass

import numpy as np
from tqdm import tqdm
from config import Config
from database import FaceDatabase
from model_engine import FaceEngine
from interface import FaceInterface
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class SmartLabeler:
    def __init__(self):
        self.config = Config()
        self.db = FaceDatabase(self.config)
        self.engine = FaceEngine(self.config)
        self.ui = FaceInterface()
        self.trained_models = {}

    def manual_fix_callback(self, face_id, new_name):
        """
        Wywoływane z GUI (FaceCard), gdy użytkownik zmieni imię
        w głównym oknie weryfikacji.
        """
        print(f"Ręczna korekta: {face_id} -> {new_name}")
        # Ustawiamy validated=True, bo użytkownik osobiście to potwierdził
        self.db.set_label(face_id, new_name, validated=True)

        # Pobieramy wszystkie twarze dla nowej osoby i douczamy model (zespalanie z GUI)
        labeled_faces = self.db.get_all_labeled_faces()
        embeddings_for_svm = []
        for fid, label in labeled_faces:
            if label == new_name:
                meta = self.db.get_metadata_for_gui(fid)
                if meta and 'embedding' in meta:
                    embeddings_for_svm.append(meta['embedding'])

        if len(embeddings_for_svm) > 0:
            self._train_identity_model(new_name, embeddings_for_svm)

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
                # self.db.mark_as_processed(full_path)  # Oznaczamy jako przetworzone

        # 4. Zamknięcie paska przed startem właściwego etykietowania
        self.ui.close_startup_progress()
        print(f"\nScan complete.")

    def process_bulk_selection(self, face_ids):
        """Obsługuje okno dialogowe DBSCAN i trenuje model SVM."""
        selected_fids, name = self.ui.bulk_verify_faces(face_ids)
        if not selected_fids or not name:
            return

        # 1. Zapisujemy nowe etykiety w bazie (tylko te wybrane ptaszkiem)
        for fid in selected_fids:
            self.db.set_label(fid, name, validated=True)

        # 2. POBIERAMY WSZYSTKIE TWARZE TEJ OSOBY Z BAZY (Zespalanie starych z nowymi)
        labeled_faces = self.db.get_all_labeled_faces()
        embeddings_for_svm = []

        for fid, label in labeled_faces:
            if label == name:
                meta = self.db.get_metadata_for_gui(fid)
                if meta and 'embedding' in meta:
                    embeddings_for_svm.append(meta['embedding'])

        # 3. Trenujemy (lub nadpisujemy) model SVM na połączonym, powiększonym zbiorze
        if len(embeddings_for_svm) > 0:
            self._train_identity_model(name, embeddings_for_svm)

        # Odświeżamy widok, żeby pokazać nowo dodane twarze
        self.refresh_main_view()

    def _train_identity_model(self, name, group_embeddings):
        """Trenuje model OCSVM dla konkretnej osoby i zapisuje go w pamięci."""
        train_embs = np.array(group_embeddings)

        # Jeśli mamy za mało danych, unikamy błędu StandardScalera
        if len(train_embs) < 2:
            print(f"Za mało próbek do wytrenowania profilu SVM dla: {name} (minimum 2)")
            return

        # 1. Normalizacja (StandardScaler)
        scaler = StandardScaler()
        train_embs_scaled = scaler.fit_transform(train_embs)

        # 2. Trening OCSVM
        clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.4)
        clf.fit(train_embs_scaled)

        # 3. Zapisanie modelu i scalera do słownika
        self.trained_models[name] = {
            'model': clf,
            'scaler': scaler
        }
        print(f"Wytrenowano profil SVM dla: {name} (na {len(train_embs)} próbkach)")

    def run_clustering_phase(self):
        """Uruchamia DBSCAN na wektorach i pokazuje użytkownikowi propozycje grup."""
        print("\n--- Rozpoczynam fazę grupowania (DBSCAN) ---")
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data: return

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        # 1. Normalizacja całej chmury punktów
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)

        # 2. PCA na znormalizowanych danych
        pca = PCA(n_components=min(50, len(embeddings)))
        reduced_embeddings = pca.fit_transform(embeddings_norm)

        # 3. Wywołanie silnika z parametrami
        clusters = self.engine.get_face_clusters(reduced_embeddings, fids)

        # Filtrujemy grupy: bierzemy tylko te, które mają np. minimum 4 zdjęcia
        valid_clusters = {cid: cfids for cid, cfids in clusters.items() if len(cfids) >= 4}

        print(f"Znaleziono {len(clusters)} grup ogółem, do weryfikacji zakwalifikowano {len(valid_clusters)}.")

        for cluster_id, cluster_fids in valid_clusters.items():
            # Wywołujemy okno tylko dla grup, które mają sens
            self.process_bulk_selection(cluster_fids)
            self.refresh_main_view()

        # Reszta (małe grupki i szum) zostanie dla SVM
        self.run_competition_phase()

        # Ostatnie odświeżenie głównego widoku
        self.refresh_main_view()

    def run_competition_phase(self):
        """Wszystkie wytrenowane modele rywalizują o niepodpisane twarze."""
        print("\n--- Rozpoczynam rywalizację modeli SVM (Argmax) ---")
        unlabeled_data = self.db.get_all_unlabeled_embeddings()

        if not unlabeled_data or not self.trained_models:
            print("Brak danych lub wytrenowanych modeli do rywalizacji.")
            return

        fids = [item[0] for item in unlabeled_data]
        test_embs = np.array([item[1] for item in unlabeled_data])

        best_matches = {fid: {'best_score': 0.0, 'best_name': None} for fid in fids}

        for name, pipeline in self.trained_models.items():
            scaler = pipeline['scaler']
            clf = pipeline['model']

            # Zabezpieczenie na wypadek błędnych wymiarów
            if test_embs.shape[1] != scaler.mean_.shape[0]:
                continue

            test_embs_scaled = scaler.transform(test_embs)
            scores = clf.decision_function(test_embs_scaled)

            # DEBUG: Sprawdźmy najwyższy wynik dla tego modelu
            if len(scores) > 0:
                print(f"Model {name}: max score = {np.max(scores):.4f}, min score = {np.min(scores):.4f}")

            for fid, score in zip(fids, scores):
                # score > 0 oznacza, że model uznaje twarz za należącą do swojej klasy
                if score > 0 and score > best_matches[fid]['best_score']:
                    best_matches[fid]['best_score'] = score
                    best_matches[fid]['best_name'] = name

        matches_count = 0
        for fid, match in best_matches.items():
            if match['best_name'] is not None:
                # validated=False oznacza, że zrobił to automat
                self.db.set_label(fid, match['best_name'], validated=False)
                matches_count += 1

        print(f"Zakończono rywalizację. Automatycznie sklasyfikowano {matches_count} twarzy.")

    def start_labeling(self):
        """Finalizuje proces i utrzymuje okno otwarte dla użytkownika."""
        print("\n--- Wszystkie fazy zakończone. Panel weryfikacji gotowy. ---")
        self.refresh_main_view()
        sys.exit(self.ui.app.exec_())


if __name__ == "__main__":
    app = SmartLabeler()
    app.run_initial_scan(limit=300)
    app.run_clustering_phase()
    app.start_labeling()