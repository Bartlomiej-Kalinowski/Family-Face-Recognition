"""GUI tool for DBSCAN-assisted ground-truth labeling of extracted faces."""

from __future__ import annotations

from ml_engine import FaceExtractor, FaceClassifier

import os
import re
import shutil
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from tqdm import tqdm

from config import Config
from database import FaceDatabase
from interface import FaceInterface


class GroundTruthClusterTool:
    """Label all unlabeled faces by iterating DBSCAN clusters and manual corrections."""

    def __init__(self, config: Config, db: FaceDatabase, min_cluster_size: int = 1):
        """Initialize tool with dependency injection for Config and DB."""
        self.config = config
        self.db = db
        self.ui = FaceInterface()
        self.classifier = FaceClassifier()

        self.min_cluster_size = min_cluster_size
        self._label_next_index: dict[str, int] = {}
        self.dataset_id: int = 1
        self.mode = "labeling"

    def _setup_session_via_gui(self) -> bool:
        """Prompt user for dataset ID and preprocessing options via UI."""
        # 1. Wybór Datasetu
        items = ["1", "2", "3"]
        item, ok = QInputDialog.getItem(
            self.ui, "Wybór Datasetu", "Na którym zestawie danych chcesz pracować?", items, 0, False
        )
        if not ok:
            return False
        self.dataset_id = int(item)

        # 2 tryb
        items = ["labeling", "change record", "delete record", "copy dataset"]
        item, ok = QInputDialog.getItem(
            self.ui, "Wybor trybu", "Wybierz tryb: ", items, 0, False
        )
        if not ok:
            return False
        self.mode = item

        # omitting next questions and pop-ups
        if self.mode == "change record" or self.mode == "copy dataset":
            return True

        # 3. Pytanie o Recompute Embeddings
        reply = QMessageBox.question(
            self.ui,
            "Przygotowanie danych",
            "Czy chcesz ponownie przeliczyć embeddingi (HOG + PCA) przed startem klastrowania?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            extractor = FaceExtractor(self.config, self.db, dataset_id=self.dataset_id)
            extractor.compute_embedding_from_crop()

        return True


    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Keep filename-safe characters for generated face IDs."""
        return re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")

    def _ensure_label_counter(self, safe_label: str) -> None:
        """Initialize next index for label based on existing rows in DB."""
        if safe_label in self._label_next_index:
            return

        self.db._cursor.execute(
            "SELECT face_id FROM faces WHERE face_id LIKE ? AND dataset_id = ?",
            (f"{safe_label}_%", self.dataset_id),
        )
        max_idx = 0
        for (face_id,) in self.db._cursor.fetchall():
            suffix = face_id[len(safe_label) + 1 :]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        self._label_next_index[safe_label] = max_idx + 1

    def _allocate_new_name(self, safe_label: str) -> tuple[str, str]:
        """Reserve a collision-free `<label>_<idx>` face ID and absolute crop path."""
        self._ensure_label_counter(safe_label)
        next_idx = self._label_next_index[safe_label]

        while True:
            candidate = f"{safe_label}_{next_idx:04d}"
            candidate_path = os.path.join(self.config.FACES_DIR, f"{candidate}.jpg")

            self.db._cursor.execute(
                "SELECT 1 FROM faces WHERE face_id = ? AND dataset_id = ?",
                (candidate, self.dataset_id)
            )
            db_has_id = self.db._cursor.fetchone() is not None
            fs_has_file = os.path.exists(candidate_path)

            if not db_has_id and not fs_has_file:
                self._label_next_index[safe_label] = next_idx + 1
                return candidate, os.path.abspath(candidate_path)

            next_idx += 1

    def _rename_face_and_sync(self, old_face_id: str, safe_label: str) -> None:
        """Safely rename the physical file and update database references."""
        new_face_id, new_face_path = self._allocate_new_name(safe_label)
        old_face_path = os.path.join(self.config.FACES_DIR, f"{old_face_id}.jpg")

        if not os.path.exists(old_face_path):
            print(f"[ERROR] Plik źródłowy nie istnieje: {old_face_path}. Pomijam aktualizację DB.")
            return  # NIE aktualizujemy bazy, jeśli nie ma pliku!

        try:
            shutil.move(old_face_path, new_face_path)
        except Exception as e:
            print(f"[ERROR] Błąd shutil.move: {e}")
            return

            # 2. Jeśli dysk się udał, aktualizujemy BAZĘ (w tym image_path!)
        self.db._cursor.execute(
            """UPDATE faces
               SET face_id            = ?,
                   image_path         = ?,
                   ground_truth_label = ?,
                   is_test            = 0
               WHERE face_id = ?
                 AND dataset_id = ?""",
            (new_face_id, new_face_path, safe_label, old_face_id, self.dataset_id)
        )
        self.db._conn.commit()

    def _get_unlabeled_data(self) -> tuple[list[str], np.ndarray]:
        """Fetch all unlabeled faces and return strictly typed IDs and Embeddings."""
        rows = self.db.get_all_unlabeled_embeddings(dataset=self.dataset_id)
        print(f"[INFO] Pobieram {len(rows)} nieetykietowanych twarzy...")
        if not rows:
            return [], np.array([])

        fids = [row[0] for row in rows]
        # Bezpieczne rzutowanie list na macierz float32 wymaganą przez Scikit-Learn
        embs = np.array([row[1] for row in rows], dtype=np.float32)
        return fids, embs

    def _assign_label(self, face_ids: list[str], label: str) -> int:
        """Rename files and persist ground-truth label for selected records."""
        clean_label = label.strip()
        safe_label = self._sanitize_label(clean_label)

        if not clean_label or not safe_label:
            return 0

        written = 0
        total = len(face_ids)
        for idx, fid in enumerate(face_ids, start=1):
            self._rename_face_and_sync(fid, safe_label)
            written += 1

            # Odświeżanie GUI co kilka iteracji lub na końcu
            if idx % 5 == 0 or idx == total:
                self.ui.update_progress(idx, total, f"Zapisywanie: {clean_label}", )
                QApplication.processEvents()

        return written

    def copy_dataset(self, src_dataset: int, dst_dataset: int):
        """Copy all faces from one dataset to another."""
        if src_dataset == dst_dataset:
            print("Chciano skopiowac z tego samego datasetu do tego samego. Zamykam program")
            return False
        read_cursor = self.db._conn.cursor()
        offset = 0
        batch = 50
        # index to avoid duplicates
        self.db._cursor.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_face_dataset ON faces (face_id, dataset_id)""")
        while True:
            read_cursor.execute(
                """SELECT face_id, original_image, image_path, bbox, embedding, ground_truth_label 
                   FROM faces WHERE dataset_id = ? AND ground_truth_label != 'None' ORDER BY face_id LIMIT ? OFFSET ?""",
                (src_dataset, batch, offset)
            )
            rows = read_cursor.fetchall()
            if not rows:
                break
            for fid, orig_image, img_path, bbox, emb,ground_truth in tqdm(rows, "Kopiowanie"):

                self.db._cursor.execute("""INSERT OR IGNORE INTO faces (face_id,  dataset_id, original_image, image_path, bbox,
                                                      embedding, manual_label, svm_prediction , ground_truth_label, is_test)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (fid, dst_dataset, orig_image, img_path, bbox, emb, None, None, ground_truth, 0))
            offset += batch
        self.db._conn.commit()
        return True

    def change_record(self, dataset: int = 1):
        old_img_path, ok = QInputDialog.getText(
            None,
            "Podaj ścieżkę",
            "Wklej ścieżkę do pliku z twarzą (zamknij okno by zakonczyc):"
        )

        if ok and old_img_path:
            print("Podana ścieżka:", old_img_path)

        else:
            print("Anulowano lub brak danych")
            exit(0)

        self.db._cursor.execute("""SELECT face_id FROM faces WHERE dataset_id = ? AND image_path = ?"""
                                , (dataset, old_img_path))

        face_id = self.db._cursor.fetchone()[0]

        new_label, ok = QInputDialog.getText(
            None,
            "Podaj nowa etykietę",
            "Wklej nową etykietę:"
        )

        if ok and new_label:
            print("Podana etykieta:", new_label)

        else:
            print("Anulowano lub brak danych")

        self._rename_face_and_sync(face_id, new_label)


    def run(self) -> None:

        """Main workflow for clustering and batch labeling."""
        print("[INFO] Start etykietowania ground truth przez DBSCAN + GUI.")

        # 1. Konfiguracja początkowa przez GUI
        if not self._setup_session_via_gui():
            print("[INFO] Operacja przerwana przez użytkownika.")
            return

        if self.mode == "change record":
            while True:
                self.change_record(self.dataset_id)

        elif self.mode == "copy dataset":
            src_dataset = self.ui.ask_for_scan_dataset_id("Wybór zestawu danych", "Wybierz zestaw źródłowy:")
            dst_dataset = self.ui.ask_for_scan_dataset_id("Wybór zestawu danych", "Wybierz zestaw docelowy:")
            if src_dataset == dst_dataset:
                QMessageBox.warning(
                    self.ui,
                    "Niepoprawny wybor",
                    "Dataset zrodlowy i docelowy musza byc rozne.",
                )
                return
            self.copy_dataset(src_dataset, dst_dataset)
            return

        else:
            # 2. Pobranie niepodpisanych danych
            fids, embeddings = self._get_unlabeled_data()
            print("Dlugosc wektora twarzy: ", len(embeddings[0]))

            if len(fids) == 0:
                QMessageBox.information(self.ui, "Gotowe", "Wszystkie twarze mają już etykiety.")
                return

            if len(fids) < self.min_cluster_size:
                QMessageBox.warning(self.ui, "Brak danych", "Za mało niepodpisanych twarzy by użyć DBSCAN.")
                return

            print(f"[INFO] Rozpoczynam klastrowanie {len(fids)} twarzy...")

            # 3. Uruchomienie DBSCAN (metoda z FaceClassifier)
            clusters_dict = self.classifier.get_face_clusters(embeddings, fids)

            # Odfiltrowanie małych klastrów i sortowanie od największych
            valid_clusters = [cfids for cid, cfids in clusters_dict.items() if len(cfids) >= self.min_cluster_size]
            print(len(valid_clusters))
            valid_clusters.sort(key=len, reverse=True)

            if not valid_clusters:
                QMessageBox.information(self.ui, "Wynik DBSCAN", "Algorytm nie znalazł żadnych wyraźnych grup.")
                return

            # 4. Przekazanie klastrów do GUI do masowego zatwierdzenia
            rounds_without_progress = 0

            to_classify = len(fids)
            for cluster_fids in valid_clusters:
                print("Liczba twarzy: ", to_classify)
                # Pobieramy pełne pary (fid, path) by UI mogło wyświetlić obrazki
                cluster_data = self.db.get_paths_for_fids(cluster_fids, dataset=self.dataset_id)

                # Wywołanie Bulk UI
                selected_fids, label_name = self.ui.bulk_verify_faces(cluster_data)

                # Obsługa przycisku "Anuluj" lub zamknięcia okna
                if selected_fids is None and label_name is None:
                    print("[INFO] Przerwano etykietowanie na żądanie użytkownika.")
                    break

                # Jeśli użytkownik coś podpisał
                if selected_fids and label_name:
                    written = self._assign_label(selected_fids, label_name)
                    to_classify  -= written
                    if written > 0:
                        rounds_without_progress = 0
                    else:
                        rounds_without_progress += 1
                else:
                    rounds_without_progress += 1

                if rounds_without_progress >= 3:
                    QMessageBox.warning(
                        self.ui,
                        "Brak postępu",
                        "Pominięto kilka grup z rzędu. Przerywam automatyczne podpowiadanie."
                    )
                    break

            print("[INFO] Zakończono sesję etykietowania.")
            QMessageBox.information(self.ui, "Koniec", "Sesja klastrowania i podpisywania zakończona.")



if __name__ == "__main__":
    # Inicjalizacja zależności przed startem logiki
    app_config = Config()
    app_db = FaceDatabase(app_config)

    tool = GroundTruthClusterTool(config=app_config, db=app_db)

    try:
        tool.run()
    finally:
        # Zawsze bezpiecznie zamykamy bazę na końcu pliku głównego
        app_db.close()