"""Application controller tying together UI, database, and ML workflows."""

from ml_engine import FaceClassifier, FaceExtractor

import hashlib
import os
import shutil
import sys
from datetime import datetime
import json

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox # only for visualisation
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from config import Config
from database import FaceDatabase
from interface import FaceInterface


class SmartLabelerController:
    """Coordinate scan, labeling, training, prediction, and visualization stages."""

    def __init__(self):
        self.config = Config()
        # 1. Tworzymy obiekt bazy danych w całej aplikacji
        self.db = FaceDatabase(self.config)
        # 2. Przekazujemy ten sam obiekt (referencję) do ekstraktora
        self.extractor = FaceExtractor(self.config, self.db)
        self.classifier = FaceClassifier()
        self.ui = FaceInterface()
        self.dataset = 1
        self.ui.set_visualize_callback(self._on_generate_visualization_clicked)

    def _manual_fix_callback(self, face_id: str, new_name: str) -> None:
        """Persist a manual label correction triggered from the UI."""
        print(f"Poprawka ręczna: {face_id} -> {new_name}")
        self.db.set_manual_label(face_id, new_name, dataset=self.dataset)
        self.refresh_main_view()

    def refresh_main_view(self) -> None:
        """Reload labeled records from the database and repopulate the UI grid."""
        labeled_faces = self.db.get_all_labeled_faces(dataset=self.dataset)
        self.ui.refresh_classified_faces(labeled_faces, self._manual_fix_callback)

    def run_initial_scan(self, mode: str, limit: int = 100000, callback=None) -> None:
        """Scan source images, extract faces, and store face crops with embeddings."""
        if mode == "full":
            self.db.clear_database(self.dataset)
            print("Clearing database...")
            for crop in os.listdir(self.config.FACES_DIR):
                os.remove(os.path.join(self.config.FACES_DIR, crop))

        all_paths = [
            os.path.normpath(os.path.abspath(os.path.join(root, f_name)))
            for root, _, files in os.walk(self.config.SOURCE_DIR)
            for f_name in files
            if f_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".heic"))
        ]

        processed_paths = set(self.db.get_all_processed_paths(dataset=self.dataset))
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
                    self.db.save_face(face["original_image"],face["crop"], fid, face["bbox"]
                                      ,self.dataset, clean_emb)
                self.db.mark_as_processed(full_path, dataset=self.dataset)
                self.db._conn.commit()

                if callback:
                    callback(i + 1, total, f"Skanowanie: {os.path.basename(full_path)}")
            except Exception as e:
                pbar.write(f"Błąd: {e}")

        total_faces = self.db.get_total_faces_count(dataset=self.dataset)
        if hasattr(self, "ui"):
            self.ui.update_face_stats(total_faces)

        print(f"Skanowanie zakończone. Wycięto i zapisano {total_faces} twarzy.")

    def process_bulk_selection(self, id_path_pairs: dict) -> None:
        """id_path_pairs to lista [(fid1, path1), (fid2, path2), ...]"""
        if not id_path_pairs:
            return

        # Przekazujemy do UI pary (ID, ścieżka), żeby okno mogło wyświetlić miniatury
        selected_fids, name = self.ui.bulk_verify_faces(id_path_pairs)

        if not selected_fids or not name:
            return

        total = len(selected_fids)
        for i, fid in enumerate(selected_fids):
            # Zapisujemy w bazie
            self.db.set_manual_label(fid, name, dataset=self.dataset, is_test=0)

            if i % 10 == 0:  # Optymalizacja odświeżania paska postępu
                self.ui.update_progress(i + 1, total, f"Zapisywanie: {name}")
                QApplication.processEvents()

        self.db._conn.commit()
        self.refresh_main_view()

    def preprocessing_phase(self):
        self.extractor.compute_embedding_from_crop()


    def run_clustering_phase(self) -> dict:
        """Run DBSCAN clustering and return whether training can continue."""
        unlabeled_data = self.db.get_all_embeddings_without_ground_truth(dataset=self.dataset)
        print("Number of unlabeled faces in database -- test:\t", len(unlabeled_data))
        if not unlabeled_data:
            print("[INFO] No unlabeled embeddings in database for clustering!")
            return {"ready_for_training": False, "labeled_count": 0}

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        clusters = self.classifier.get_face_clusters(embeddings, fids) # dict of {"label1": [fid1, fid2, ...], ...}

        # cid - cluster id - label
        # cfids - list of fids
        # valid clusters is clusters without too small clusters
        valid_clusters = {cid: cfids for cid, cfids in clusters.items() if len(cfids) >= 3}

        print("Number of all DBSCAN clusters:\t", len(clusters))
        print("Valid clusters number (more than 3 faces per one cluster):\t", len(valid_clusters))

        for cluster_fids in valid_clusters.values():
            print("Number of faces in cluster:\t", len(cluster_fids))

            # Pobieramy ścieżki do zdjęć dla tego klastra, żeby GUI mogło je wyświetlić
            cluster_fids_and_paths = self.db.get_paths_for_fids(cluster_fids, dataset=self.dataset)

            # Przekazujemy pary do funkcji bulk
            self.process_bulk_selection(cluster_fids_and_paths)

        # number of manual labels after DBSCAN
        labeled_count = len(set(label for _, label, _ in self.db.get_labeled_data_for_train(dataset=self.dataset)))
        ready_for_training = labeled_count >= 2
        if not ready_for_training:
            print("Too less train different labels (min. 2 required), to start SVM prediction!")
        return {"ready_for_training": ready_for_training, "labeled_count": labeled_count}

    def run_classification_phase(self):
        """Train the SVM pipeline and return train samples for evaluation."""
        print("\n[SYSTEM] Starting train phase...")

        self.db.mark_unlabeled_as_test(dataset=self.dataset) # unlabeled data as test data

        train_data = self.db.get_labeled_data_for_train(dataset=self.dataset)
        if not train_data:
            print("[BŁĄD] Brak danych treningowych w bazie.")
            return None

        unique_labels = set(label for _, label, _ in train_data)
        if len(unique_labels) < 2:
            print(f"[BŁĄD] Zbyt mało osób ({len(unique_labels)}). Potrzeba min. 2 do SVM.")
            return None

        _, train_labels, train_embs = zip(*train_data)
        self.classifier.train_one_vs_rest_svm(list(train_embs), list(train_labels))
        return train_data

    def run_evaluation_phase(self, train_data):
        """Run predictions for test data, log metrics, and refresh UI with results."""
        test_data = self.db.get_unlabeled_test_data(dataset=self.dataset)

        if not test_data:
            print("[INFO] Brak nowych danych testowych do klasyfikacji.")
            return False

        fids, paths, test_embs, _ = zip(*test_data)

        y_pred, confidences = self.classifier.predict_unlabeled(list(test_embs))
        if len(y_pred) == 0:
            print("[BŁĄD] Model nie zwrócił predykcji.")
            return False

        y_true = [self.db.get_gt_from_path(path) for path in paths]

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
            self.db.set_svm_prediction(fid, pred, dataset=self.dataset)

        classified_list = list(zip(fids, y_pred))
        self.ui.refresh_classified_faces(classified_list, self._manual_fix_callback)
        self.ui.set_visualization_enabled(True)
        QMessageBox.information(
            self.ui,
            "Weryfikacja etykiet",
            "Możesz teraz poprawić etykiety ręcznie w kafelkach.\n"
            "Gdy skończysz, kliknij przycisk 'Generuj wizualizacje'.",
        )
        return True

    def app_pipeline(self) -> None:
        """Manage clustering -> classification -> evaluation using explicit returns."""
        #-------------clustering and labeling by user------------------------
        clustering_result = self.run_clustering_phase()
        if not clustering_result["ready_for_training"]:
            return
        print(f"Have {clustering_result['labeled_count']} people labeled by user. Starting SVM prediction...")

        #-------------classification-train phase------------------------------
        train_data = self.run_classification_phase()
        if not train_data:
            return

        # ------------evaluation phase - test --------------------------------
        self.run_evaluation_phase(train_data)

    def _on_generate_visualization_clicked(self) -> None:
        """Generate annotated images after optional manual corrections in the grid."""
        reply = self.ui.confirm_all_labels()
        if reply != 0:
            return

        self.ui.set_visualization_enabled(False)
        try:
            self.draw_all_labels_on_faces(self.config.ANNOTATED_FACES_DIR)
        finally:
            self.ui.set_visualization_enabled(True)

    def start(self) -> None:
        """Start the application workflow based on the selected scan mode."""
        self.dataset = self.ui.ask_for_scan_dataset_id()
        if self.dataset == -1:
            print("Kończę działanie aplikacji...")
            return
        mode = self.ui.ask_for_scan_mode()

        if mode == "use_existing":
            reply = QMessageBox.question(
                self.ui,
                "Przygotowanie danych",
                "Czy chcesz ponownie przeliczyć embeddingi (HOG + PCA) przed startem klastrowania?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.preprocessing_phase()
            self.db.rebuild_db_from_files(dataset=self.dataset)
            self.app_pipeline()
        elif mode == "full":
            for img in os.listdir(self.config.ANNOTATED_FACES_DIR):
                os.remove(os.path.join(self.config.ANNOTATED_FACES_DIR, img))
            self.run_initial_scan(mode=mode, limit=100000)
            self.preprocessing_phase()
            self.app_pipeline()
        elif mode == "incremental":
            self.run_initial_scan(mode=mode, limit=100000)
            self.preprocessing_phase()
            self.app_pipeline()
        elif mode == "cancel":
            return

        self.refresh_main_view()
        sys.exit(self.ui.app.exec_())


    def _get_visualization_path(self, original_image_path: str, target_dir: str) -> str:
        """Build a collision-safe output path for an annotated source image."""
        original_abs = os.path.abspath(original_image_path)
        source_abs = os.path.abspath(self.config.SOURCE_DIR)

        try:
            rel_path = os.path.relpath(original_abs, source_abs)
        except ValueError:
            rel_path = ""

        if rel_path and not rel_path.startswith("..") and not os.path.isabs(rel_path):
            target_path = os.path.join(target_dir, rel_path)
        else:
            digest = hashlib.sha1(original_abs.encode("utf-8")).hexdigest()[:12]
            target_path = os.path.join(target_dir, f"{digest}_{os.path.basename(original_abs)}")

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        return target_path

    @staticmethod
    def _parse_bbox(bbox_raw):
        """Parse bbox from DB and normalize to integer XYXY order."""
        if bbox_raw is None:
            return None

        if isinstance(bbox_raw, str):
            bbox = json.loads(bbox_raw)
        else:
            bbox = bbox_raw

        if isinstance(bbox, dict):
            keys = ("x1", "y1", "x2", "y2")
            if not all(k in bbox for k in keys):
                return None
            values = [bbox[k] for k in keys]
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            values = list(bbox)
        else:
            return None

        x1, y1, x2, y2 = [int(round(float(v))) for v in values]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def draw_all_labels_on_faces(self, target_dir):
        """Draw labels on face images and save visualized results to `target_dir`."""
        print("[SYSTEM] Generowanie boksów i etykiet na wszystkich twarzach...")

        results = self.db.get_all_labeled_faces(dataset=self.dataset)

        for original_image_path, _face_id, label, manual_label, _source_image_path, bbox_raw in results:
            is_manual = manual_label is not None
            visualized_path = self._get_visualization_path(original_image_path, target_dir)

            if not os.path.exists(visualized_path):
                shutil.copy2(original_image_path, visualized_path)

            img = cv2.imread(visualized_path)
            if img is None:
                continue

            h, w, _ = img.shape

            box_color = (60, 170, 75) if bool(is_manual) else (200, 140, 70)
            label_bg = (26, 26, 26)
            text_color = (245, 245, 245)
            thickness = max(1, min(3, int(min(h, w) / 450)))
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw stored YOLO bbox on the original image.
            try:
                parsed_bbox = self._parse_bbox(bbox_raw)
                if parsed_bbox is None:
                    continue
                x1, y1, x2, y2 = parsed_bbox
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox_height = max(1, y2 - y1)
            font_scale = max(0.35, min(0.62, bbox_height / 230))
            font_thickness = max(1, int(round(font_scale * 2)))

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

            source_tag = "[MANUAL]" if bool(is_manual) else "[SVM]"
            label_text = f"{source_tag} {str(label or 'Unknown').upper()}"

            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            label_x = x1
            label_y = max(lbl_h + 10, y1)
            cv2.rectangle(
                img,
                (label_x, label_y - lbl_h - 10),
                (min(label_x + lbl_w + 10, w - 1), label_y),
                label_bg,
                -1,
            )
            cv2.rectangle(
                img,
                (label_x, label_y - lbl_h - 10),
                (min(label_x + lbl_w + 10, w - 1), label_y),
                box_color,
                1,
            )
            cv2.putText(
                img,
                label_text,
                (label_x + 5, label_y - 5),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

            cv2.imwrite(visualized_path, img)

        QMessageBox.information(None, "Sukces", f"Zapisano wizualizacje w:\n{target_dir}")
        print(f"[SUKCES] Folder '{target_dir}' został zaktualizowany o ramki.")


if __name__ == "__main__":
    app = SmartLabelerController()
    app.start()
