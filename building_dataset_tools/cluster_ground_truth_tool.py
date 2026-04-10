"""GUI tool for DBSCAN-assisted ground-truth labeling of extracted faces."""

from __future__ import annotations

from ml_engine import FaceExtractor, FaceClassifier

import os
import re
import shutil
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog

from config import Config
from database import FaceDatabase
from interface import FaceInterface


class GroundTruthClusterTool:
    """Label all unlabeled faces by iterating DBSCAN clusters and manual corrections."""

    def __init__(self, config: Config, db: FaceDatabase, min_cluster_size: int = 3):
        """Initialize tool with dependency injection for Config and DB."""
        self.config = config
        self.db = db
        self.ui = FaceInterface()
        self.classifier = FaceClassifier()

        self.min_cluster_size = max(2, int(min_cluster_size))
        self._label_next_index: dict[str, int] = {}
        self.dataset_id: int = 1

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

        # 2. Pytanie o Recompute Embeddings
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

        # 1. Zmiana nazwy pliku na dysku (jeśli istnieje)
        if os.path.exists(old_face_path):
            shutil.move(old_face_path, new_face_path)

        # 2. Aktualizacja rekordu w bazie danych
        # Zakładam standardowe kolumny na podstawie Twoich poprzednich skryptów
        self.db._cursor.execute(
            """UPDATE faces 
               SET face_id = ?, manual_label = ?, is_test = 0 
               WHERE face_id = ? AND dataset_id = ?""",
            (new_face_id, safe_label, old_face_id, self.dataset_id)
        )
        self.db._conn.commit()

    def _get_unlabeled_data(self) -> tuple[list[str], np.ndarray]:
        """Fetch all unlabeled faces and return strictly typed IDs and Embeddings."""
        rows = self.db.get_all_embeddings_without_ground_truth(dataset=self.dataset_id)
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
                self.ui.update_progress(idx, total, f"Zapisywanie: {clean_label}")
                QApplication.processEvents()

        return written

    def run(self) -> None:
        """Main workflow for clustering and batch labeling."""
        print("[INFO] Start etykietowania ground truth przez DBSCAN + GUI.")

        # 1. Konfiguracja początkowa przez GUI
        if not self._setup_session_via_gui():
            print("[INFO] Operacja przerwana przez użytkownika.")
            return

        # 2. Pobranie niepodpisanych danych
        fids, embeddings = self._get_unlabeled_data()

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
        valid_clusters.sort(key=len, reverse=True)

        if not valid_clusters:
            QMessageBox.information(self.ui, "Wynik DBSCAN", "Algorytm nie znalazł żadnych wyraźnych grup.")
            return

        # 4. Przekazanie klastrów do GUI do masowego zatwierdzenia
        rounds_without_progress = 0

        for cluster_fids in valid_clusters:
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