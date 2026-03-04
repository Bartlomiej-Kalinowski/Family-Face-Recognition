import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Wymuszamy załadowanie silnika YOLO (torch) ZANIM PyQt5 przejmie kontrolę
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

        os.makedirs(self.config.FACES_DIR, exist_ok=True)
        print(f"Directory with faces ready : {os.path.abspath(self.config.FACES_DIR)}")

        self.db = FaceDatabase(self.config)

        self.engine = FaceEngine(self.config)
        self.ui = FaceInterface()
        self.trained_models = {}

    def manual_fix_callback(self, face_id, new_name):
        """zapisuje etykietę wybraną przez użytkownika."""
        print(f"Ręczna etykieta: {face_id} -> {new_name}")

        # standardowy set_label
        self.db.set_label(face_id, new_name)

        # Odświeżamy widok
        self.refresh_main_view()

    def refresh_main_view(self):
        """Pobiera dane z bazy i odświeża siatkę w głównym oknie."""
        labeled_faces = self.db.get_all_labeled_faces()
        # Przekazujemy listę krotek i metodę callback do obsługi zmian
        self.ui.refresh_classified_faces(labeled_faces, self.manual_fix_callback)

    def run_initial_scan(self, mode="incremental", limit=1000):
        """
        Tylko wycinanie twarzy + stabilny zapis.
        Zachowano strukturę pozwalającą na wizualizację klasyfikacji w przyszłości.
        """
        # 1. Przygotowanie folderów
        os.makedirs(self.config.FACES_DIR, exist_ok=True)
        # Folder na wyniki wizualizacji (np. zdjęcia z ramkami)
        output_viz_dir = os.path.join(self.config.OUTPUT_DIR, "visualizations")
        os.makedirs(output_viz_dir, exist_ok=True)

        if mode == "full":
            print("Tryb FULL: Czyszczenie bazy i folderu twarzy...")
            self.db.clear_database()
            for f in os.listdir(self.config.FACES_DIR):
                os.remove(os.path.join(self.config.FACES_DIR, f))

        # 2. Zbieranie plików
        all_paths = []
        for root, _, files in os.walk(self.config.SOURCE_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.abspath(os.path.join(root, file)))

        # 3. Filtrowanie już przetworzonych
        processed = set(self.db.get_all_processed_paths())
        to_process = [p for p in all_paths if p not in processed][:limit]

        if not to_process:
            print("Brak nowych zdjęć do przetworzenia.")
            return

        # 4. Główna pętla YOLO
        pbar = tqdm(to_process, desc="Ekstrakcja twarzy")
        for full_path in pbar:
            try:
                detected_faces = self.engine.extract_face_data(full_path)
                file_name = os.path.basename(full_path)

                for i, face in enumerate(detected_faces):
                    # Czyścimy nazwę pliku: usuwamy spacje i kropki
                    safe_name = os.path.basename(full_path).replace(" ", "_").replace(".", "_")
                    face_id = f"{safe_name}_f{i}"

                    self.db.save_face(face['crop'], face_id, full_path, face['bbox'])
                    self.db.update_embedding(face_id, face['embedding'])

                self.db.mark_as_processed(full_path)

            except Exception as e:
                pbar.write(f"Błąd pliku {file_name}: {e}")

        print(f"Zakończono. Przetworzono {len(to_process)} zdjęć.")

    def process_bulk_selection(self, face_ids):
        """
        Przywrócona wersja tradycyjna:
        Zapisuje etykiety wybrane przez użytkownika i odświeża widok.
        """
        # Wyświetlamy okno z twarzami z DBSCAN
        selected_fids, name = self.ui.bulk_verify_faces(face_ids)

        # Jeśli użytkownik zamknął okno lub nie podał imienia - przerywamy
        if not selected_fids or not name:
            return

        # 1. Zapisujemy wybrane twarze w bazie danych
        for fid in selected_fids:
            # Usuwamy 'validated=True', jeśli cofnąłeś zmiany w schemacie bazy
            # Jeśli zostawiłeś kolumnę 'validated', możesz ją zostawić.
            self.db.set_label(fid, name)

        print(f"Ręcznie zaetykietowano {len(selected_fids)} twarzy jako: {name}")

        # 2. Odświeżamy tylko widok w GUI (bez trenowania SVM w locie)
        self.refresh_main_view()

    def _train_identity_model(self, name, group_embeddings):
        """Trenuje model OCSVM dla konkretnej osoby i zapisuje go w pamięci."""
        train_embs = np.array(group_embeddings)

        # Jeśli mamy za mało danych, unikamy błędu StandardScalera
        if len(train_embs) < 2:
            print(f"Za mało próbek do wytrenowania profilu SVM dla: {name} (minimum 2)")
            return

        # # 1. Normalizacja (StandardScaler)
        # scaler = StandardScaler()
        # train_embs_scaled = scaler.fit_transform(train_embs)

        # 2. Trening OCSVM
        clf = OneClassSVM(kernel='rbf', gamma="scale", nu=0.05) # gamma = "auto"
        clf.fit(train_embs)

        # 3. Zapisanie modelu i scalera do słownika
        self.trained_models[name] = {
            'clf': clf,
            'scaler': None  # Oznaczamy, że nie używamy scalera
        }
        print(f"Wytrenowano profil SVM dla: {name} (na {len(train_embs)} próbkach)")

    def run_clustering_phase(self):
        """Uruchamia DBSCAN i pokazuje użytkownikowi tylko próbki (max 10) do nauki SVM."""
        print("\n--- Rozpoczynam fazę grupowania (DBSCAN) ---")
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data: return

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        # Normalizacja i PCA (bez zmian)
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        pca = PCA(n_components=min(50, len(embeddings)))
        reduced_embeddings = pca.fit_transform(embeddings_norm)

        # Pobieramy grupy z silnika
        clusters = self.engine.get_face_clusters(reduced_embeddings, fids)

        # 1. LIMIT MINIMALNY: Odrzucamy grupy mniejsze niż 5 zdjęć (szum)
        valid_clusters = {cid: cfids for cid, cfids in clusters.items() if len(cfids) >= 3}

        print(
            f"DBSCAN znalazł {len(clusters)} grup. Do weryfikacji (treningu) wybrano {len(valid_clusters)} najsilniejszych.")

        for cluster_id, cluster_fids in valid_clusters.items():
            # 2. LIMIT MAKSYMALNY: Bierzemy tylko pierwsze 10 zdjęć!
            # Jeśli DBSCAN znalazł 40 twarzy Kamila, Ty ocenisz tylko 10 z nich.
            # Pozostałe 30 zostaje na razie bez etykiety.
            sample_fids = cluster_fids[:13]

            # Pokazujemy okno tylko dla tych 10 zdjęć
            self.process_bulk_selection(sample_fids)

        # 3. Faza SVM:
        # Teraz SVM uczy się na Twoich zatwierdzonych próbkach (max 10 na osobę).
        # Następnie skanuje WSZYSTKIE niepodpisane zdjęcia, wliczając w to
        # te odrzucone "nadwyżki" z DBSCAN i klasyfikuje je samodzielnie!
        self.run_competition_phase()

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

        # WAŻNE: Startujemy od -1.0, bo Twoje wyniki są ujemne!
        # Jeśli zostawisz 0.0, żaden wynik -0.5 nigdy nie zostanie wybrany.
        best_matches = {fid: {'best_score': -1.0, 'best_name': None} for fid in fids}

        for name, model_bundle in self.trained_models.items():
            clf = model_bundle['clf']  # Poprawiono klucz z 'model' na 'clf'

            # Używamy surowych test_embs (bez skalowania)
            scores = clf.decision_function(test_embs)

            if len(scores) > 0:
                print(f"Model {name}: max score = {np.max(scores):.4f}")

            for i, score in enumerate(scores):
                fid = fids[i]
                # Próg -0.65 jest OK, ale musi być wyższy niż startowe -1.0
                if score > -0.65 and score > best_matches[fid]['best_score']:
                    best_matches[fid]['best_score'] = score
                    best_matches[fid]['best_name'] = name

        matches_count = 0
        for fid, match in best_matches.items():
            if match['best_name'] is not None:
                self.db.set_label(fid, match['best_name'], validated=False)
                matches_count += 1

        print(f"Zakończono rywalizację. Automatycznie sklasyfikowano {matches_count} twarzy")

    def start_labeling(self):
        """Finalizuje proces i utrzymuje okno otwarte dla użytkownika."""
        print("\n--- Wszystkie fazy zakończone. Panel weryfikacji gotowy. ---")
        self.refresh_main_view()
        sys.exit(self.ui.app.exec_())

    def visualize_results(self):
        """
        Rysuje ramki i podpisy na kopiach oryginalnych zdjęć
        na podstawie aktualnych etykiet w bazie.
        """
        import cv2
        output_dir = os.path.join(self.config.OUTPUT_DIR, "visualizations")
        os.makedirs(output_dir, exist_ok=True)

        # Pobierz wszystkie twarze, które mają już jakąś etykietę
        labeled_faces = self.db.get_all_labeled_faces()  # Zwraca (face_id, label)

        # Grupowanie według ścieżki do oryginalnego zdjęcia
        image_map = {}
        for fid, label in labeled_faces:
            meta = self.db.get_metadata_for_gui(fid)  # Zakładam, że zwraca image_path i bbox
            path = meta['image_path']
            if path not in image_map: image_map[path] = []
            image_map[path].append({'label': label, 'bbox': meta['bbox']})

        print(f"Generowanie wizualizacji dla {len(image_map)} zdjęć...")
        for img_path, detections in image_map.items():
            img = cv2.imread(img_path)
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                # Rysowanie ramki
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Rysowanie etykiety
                cv2.putText(img, str(det['label']), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            save_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    app_instance = SmartLabeler()

    # 1. FRONTEND: Zapytaj użytkownika o tryb
    mode = app_instance.ui.ask_for_scan_mode()

    # 2. BACKEND: Uruchom skanowanie z wybranym trybem
    if mode != "cancel":
        app_instance.run_initial_scan(mode=mode, limit=500)
        app_instance.run_clustering_phase()
        app_instance.start_labeling()