"""Application controller tying together UI, database, and ML workflows."""

import json
import os
import shutil
import sys
from datetime import datetime

# Load torch/ultralytics early to avoid runtime initialization issues with Qt event loop.
try:
    from ultralytics import YOLO
    import torch

    _ = torch.empty(1)
except Exception:
    print("Blad importu biblioteki ultralytics lub pytorch")
    exit(0)

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from config import Config
from database import FaceDatabase
from interface import FaceInterface
from ml_engine import FaceClassifier, FaceExtractor


class SmartLabelerController:
    """Coordinate scan, labeling, training, prediction, and visualization stages."""

    def __init__(self):
        """Initialize core services and UI for the application session."""
        self.config = Config()
        self.db = FaceDatabase(self.config)
        self.extractor = FaceExtractor(self.config)
        self.classifier = FaceClassifier()
        self.ui = FaceInterface()

    def _manual_fix_callback(self, face_id: str, new_name: str) -> None:
        """Persist a manual label correction triggered from the UI."""
        print(f"Poprawka ręczna: {face_id} -> {new_name}")
        self.db.set_manual_label(face_id, new_name)
        self.refresh_main_view()

    def refresh_main_view(self) -> None:
        """Reload labeled records from the database and repopulate the UI grid."""
        labeled_faces = self.db.get_all_labeled_faces()
        self.ui.refresh_classified_faces(labeled_faces, self._manual_fix_callback)

    def run_initial_scan(self, mode: str = "incremental", limit: int = 1000, callback=None) -> None:
        """Scan source images, extract faces, and store face crops with embeddings."""
        if mode == "full":
            self.db.clear_database()
            if os.path.exists(self.config.FACES_DIR):
                for f_name in os.listdir(self.config.FACES_DIR):
                    os.remove(os.path.join(self.config.FACES_DIR, f_name))

        all_paths = [
            os.path.normpath(os.path.abspath(os.path.join(root, f_name)))
            for root, _, files in os.walk(self.config.SOURCE_DIR)
            for f_name in files
            if f_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        processed_paths = set(self.db.get_all_processed_paths())
        to_process = [path for path in all_paths if path not in processed_paths][:limit]

        if not to_process:
            print("Brak nowych zdjęć do ekstrakcji.")
            return

        pbar = tqdm(to_process, desc="Ekstrakcja YOLO")
        total = len(to_process)

        for i, full_path in enumerate(pbar):
            safe_name = os.path.basename(full_path).replace(" ", "_").replace(".", "_")
            try:
                if hasattr(self, "ui") and self.ui:
                    self.ui.update_progress(i + 1, total, f"Skanowanie: {os.path.basename(full_path)}")

                QApplication.processEvents()

                detected_faces = self.extractor.extract_face_data(full_path)

                for j, face in enumerate(detected_faces):
                    fid = f"{safe_name}_f{j}"
                    clean_emb = np.array(face["embedding"]).flatten().astype(np.float32).tolist()
                    self.db.save_face(face["crop"], fid, full_path, face["bbox"], clean_emb)

                self.db.mark_as_processed(full_path)
                self.db._conn.commit()

                if callback:
                    callback(i + 1, total, f"Skanowanie: {os.path.basename(full_path)}")
            except Exception as e:
                pbar.write(f"Błąd: {e}")

        total_faces = self.db.get_total_faces_count()
        if hasattr(self, "ui"):
            self.ui.update_face_stats(total_faces)

        print(f"Skanowanie zakończone. Wycięto i zapisano {total_faces} twarzy.")

    def process_bulk_selection(self, face_ids: list) -> None:
        """Apply one confirmed label to a selected subset from a detected cluster."""
        selected_fids, name = self.ui.bulk_verify_faces(face_ids)
        if not selected_fids or not name:
            return

        total = len(selected_fids)
        for i, fid in enumerate(selected_fids):
            self.db.set_manual_label(fid, name, is_test=0)
            self.ui.update_progress(i + 1, total, f"Zapisywanie: {name}")
            QApplication.processEvents()

        self.db._conn.commit()

    def run_clustering_phase(self) -> None:
        """Run DBSCAN clustering on unlabeled embeddings and collect human labels."""
        unlabeled_data = self.db.get_all_unlabeled_embeddings()
        if not unlabeled_data:
            return

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        clusters = self.classifier.get_face_clusters(embeddings, fids)
        valid_clusters = {cid: cfids for cid, cfids in clusters.items() if len(cfids) >= 3}

        print("Valid clusters number:\t", len(clusters))
        print("Valid clusters number:\t", len(valid_clusters))

        for cluster_fids in valid_clusters.values():
            # Limit cluster preview size so review dialogs remain usable.
            self.process_bulk_selection(cluster_fids[:13])

        labeled_count = len(set(label for _, label, _ in self.db.get_labeled_data_for_train()))

        if labeled_count >= 2:
            print(f"Mamy {labeled_count} osoby. Uruchamiam SVM...")
            self.run_classification_phase()
        else:
            print("Zbyt mało osób w bazie (wymagane min. 2), aby uruchomić SVM.")

    def run_classification_phase(self):
        """Train the SVM pipeline on manually labeled samples."""
        print("\n[SYSTEM] Rozpoczynanie fazy treningu...")

        self.db.mark_unlabeled_as_test()

        train_data = self.db.get_labeled_data_for_train()
        if not train_data:
            print("[BŁĄD] Brak danych treningowych w bazie.")
            return

        unique_labels = set(label for _, label, _ in train_data)
        if len(unique_labels) < 2:
            print(f"[BŁĄD] Zbyt mało osób ({len(unique_labels)}). Potrzeba min. 2 do SVM.")
            return

        _, train_labels, train_embs = zip(*train_data)
        self.classifier.train_multiclass_svm(list(train_embs), list(train_labels))

        self.run_evaluation_phase(train_data)

    def run_evaluation_phase(self, train_data):
        """Run predictions for test data, log metrics, and refresh UI with results."""
        test_data = self.db.get_unlabeled_test_data()

        if not test_data:
            print("[INFO] Brak nowych danych testowych do klasyfikacji.")
            return

        fids, paths, test_embs, _ = zip(*test_data)

        y_pred, confidences = self.classifier.predict_unlabeled(list(test_embs))
        if len(y_pred) == 0:
            print("[BŁĄD] Model nie zwrócił predykcji.")
            return

        y_true = [self.get_gt_from_path(path) for path in paths]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)

        log_path = "wyniki_klasyfikacji.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a", encoding="utf-8") as file:
            file.write(f"\n{'=' * 60}\n")
            file.write(f"SESJA: {timestamp}\n")
            file.write(f"Próbki: Trening={len(train_data)}, Test={len(test_data)}\n")
            file.write(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\n")
            file.write("-" * 30 + "\n")
            file.write("Raport szczegółowy:\n")
            file.write(report)
            file.write(f"{'=' * 60}\n")

        print(f"\n[SUKCES] Raport zapisany w: {log_path}")
        print(report)

        # Keep confidence scores available for future thresholding logic.
        _ = confidences

        for fid, pred in zip(fids, y_pred):
            self.db.set_svm_prediction(fid, pred)

        classified_list = list(zip(fids, y_pred))
        self.ui.refresh_classified_faces(classified_list, self._manual_fix_callback)

        reply = self.ui.confirm_all_labels()
        if reply == 0:
            self.draw_all_labels_on_faces(self.config.ANNOTATED_FACES_DIR)

    def start(self) -> None:
        """Start the application workflow based on the selected scan mode."""
        mode = self.ui.ask_for_scan_mode()

        if mode == "use_existing":
            self.rebuild_db_from_files()
            self.run_clustering_phase()
        elif mode in ["full", "incremental"]:
            self.run_initial_scan(mode=mode, limit=500)
            self.run_clustering_phase()
        elif mode == "cancel":
            return

        self.refresh_main_view()
        sys.exit(self.ui.app.exec_())

    def get_gt_from_path(self, path):
        """Extract expected label from filename convention `name_number.ext`."""
        filename = os.path.basename(path)
        name_part = os.path.splitext(filename)[0]
        parts = name_part.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return name_part

    def rebuild_db_from_files(self):
        """Rebuild face records by importing files from the extracted-face directory."""
        print("Rozpoczynam odbudowę bazy na podstawie istniejących wycinków...")

        self.db.clear_database()

        face_folder = self.config.FACES_DIR
        files = [name for name in os.listdir(face_folder) if name.endswith((".jpg", ".jpeg", ".png"))]

        if not files:
            print("Folder z twarzami jest pusty!")
            return

        total = len(files)
        for i, filename in enumerate(files):
            full_path = os.path.join(face_folder, filename)
            face_img = cv2.imread(full_path)

            if face_img is None:
                continue

            gt_label = self.get_gt_from_path(full_path)

            gray = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            embedding = np.array(hog.compute(gray)).flatten().astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            face_id = os.path.splitext(filename)[0]
            emb_json = json.dumps(embedding.tolist())
            self.db._cursor.execute(
                """
                INSERT INTO faces (face_id, image_path, embedding, is_test)
                VALUES (?, ?, ?, ?)
                """,
                (face_id, full_path, emb_json, 0),
            )

            if hasattr(self, "ui"):
                self.ui.update_progress(i + 1, total, f"Import: {gt_label}")
                QApplication.processEvents()

        self.db._conn.commit()
        print(f"Baza odbudowana. Zaimportowano {total} twarzy.")

    def draw_all_labels_on_faces(self, target_dir):
        """Draw labels on face images and save visualized results to `target_dir`."""
        print("[SYSTEM] Generowanie boksów i etykiet na wszystkich twarzach...")

        results = self.db.get_all_labeled_faces()

        for face_id, label, is_manual, original_img_path in results:
            img_path = os.path.join(target_dir, f"{face_id}.jpg")

            if not os.path.exists(img_path):
                shutil.copy2(original_img_path, img_path)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape

            color = (0, 255, 0) if is_manual else (0, 165, 255)
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, w / 300)

            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)

            source_tag = "[MANUAL]" if is_manual else "[SVM]"
            label_text = f"{source_tag} {label.upper()}"

            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font, font_scale, 1)
            cv2.rectangle(img, (0, 0), (lbl_w + 10, lbl_h + 15), color, -1)
            cv2.putText(img, label_text, (5, lbl_h + 8), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(img_path, img)

        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.information(None, "Sukces", f"Zapisano wizualizacje w:\n{target_dir}")
        print(f"[SUKCES] Folder '{target_dir}' został zaktualizowany o ramki.")


if __name__ == "__main__":
    app = SmartLabelerController()
    app.start()
