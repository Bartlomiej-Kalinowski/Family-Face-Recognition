"""GUI tool for DBSCAN-assisted ground-truth labeling of extracted faces."""

from __future__ import annotations

from ml_engine import FaceExtractor, FaceClusterer, FacePreprocessor

import os
import re
import shutil
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QLabel, QCheckBox, \
    QFrame, QScrollArea, QWidget, QGridLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
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
        self.preprocessor = None
        self.classifier = FaceClusterer()

        self.min_cluster_size = min_cluster_size
        self._label_next_index: dict[str, int] = {}
        self.mode = "labeling"
        self.dataset_id: int = 1

    def _setup_session_via_gui(self) -> bool:
        """Prompt user for dataset ID and preprocessing options via UI."""

        #wybor trybu
        items = ["labeling", "change record", "delete record", "copy dataset", "recalculate face vectors", "align faces"]
        item, ok = QInputDialog.getItem(
            self.ui, "Wybor trybu", "Wybierz tryb: ", items, 0, False
        )
        if not ok:
            return False
        self.mode = item

        # w przypadku kopiowanie wybor datasetu nie konieczny
        if self.mode == "copy dataset":
            return True

        # WybĂłr Datasetu
        items = ["1", "2", "3"]
        item, ok = QInputDialog.getItem(
            self.ui, "Wybor Datasetu", "Na ktorym zestawie danych chcesz pracowal?", items, 0, False
        )
        if not ok:
            return False
        self.dataset_id = int(item)



        # omitting next questions and pop-ups
        if self.mode == "change record"  or self.mode == "delete record" or self.mode == "align faces":
            return True

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
            print(f"[ERROR] Plik zrodlowy nie istnieje: {old_face_path}. Pomijam aktualizacje DB.")
            return  # NIE aktualizujemy bazy, jeĹ›li nie ma pliku!

        try:
            shutil.move(old_face_path, new_face_path)
        except Exception as e:
            print(f"[ERROR] Blad shutil.move: {e}")
            return

        self.db._cursor.execute(
            """UPDATE faces
               SET face_id            = ?,
                   image_path         = ?,
                   ground_truth_label = ?,
                   is_test            = 0
               WHERE face_id = ? AND dataset_id = ?""",
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
        # Bezpieczne rzutowanie list na macierz float32 wymaganÄ… przez Scikit-Learn
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

            # Odswiezanie GUI co kilka iteracji lub na koncu
            if idx % 5 == 0 or idx == total:
                self.ui.update_progress(idx, total, f"Zapisywanie: {clean_label}", )
                QApplication.processEvents()

        return written

    def copy_dataset(self, src_dataset: int, dst_dataset: int):
        """Copy all faces from one dataset to another."""
        if src_dataset == dst_dataset:
            print("Chciano skopiowac z tego samego datasetu do tego samego. Zamykam program")
            return False

        new_faces_dir = self.config.FACES_DIR + '_' + str(dst_dataset)

        #czyszczenie zbiory dst
        print(f"Czyszczenie folderu: {new_faces_dir}")
        try:
            # Usuwamy całą zawartość folderu (pliki), ale zostawiamy sam folder
            for filename in os.listdir(new_faces_dir):
                file_path = os.path.join(new_faces_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"[BŁĄD] Nie udało się wyczyścić plików: {e}")
        if not os.path.exists(new_faces_dir):
            print("Niepoprawna sciezka do katalogu z twarzami. Zamykam program")
            print(new_faces_dir)
            exit(0)

        print(f"[INFO] Usuwanie starych rekordów z datasetu {dst_dataset}...")
        self.db._cursor.execute("DELETE FROM faces WHERE dataset_id = ?", (dst_dataset,))
        self.db._conn.commit()

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
                new_img_path = img_path  # Domyślnie stara ścieżka

                if img_path and os.path.exists(img_path):
                    file_name = os.path.basename(img_path)
                    new_img_path = os.path.abspath(os.path.join(new_faces_dir, file_name))

                    # Jeśli plik w nowym miejscu jeszcze nie istnieje - kopiuje
                    if not os.path.exists(new_img_path):
                        shutil.copy2(img_path, new_img_path)

                self.db._cursor.execute("""INSERT OR IGNORE INTO faces (face_id,  dataset_id, original_image, image_path, bbox,
                                                      embedding, manual_label, svm_prediction , ground_truth_label, is_test)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (fid, dst_dataset, orig_image, new_img_path, bbox, emb, None, None, ground_truth, 0))
            offset += batch
        self.db._conn.commit()
        return True

    def delete_records(self, dataset: int):
        """Enables deleting particular records from the database."""
        read_cursor = self.db._conn.cursor()
        batch = 50
        last_face_id = None # last face id from the previous batch

        while True:
            if last_face_id is None:
                read_cursor.execute(
                    """SELECT face_id, dataset_id, image_path
                       FROM faces
                       WHERE dataset_id = ? AND ground_truth_label != 'None'
                       ORDER BY face_id LIMIT ?""",
                    (dataset, batch)
                )
            else:
                read_cursor.execute(
                    """SELECT face_id, dataset_id, image_path
                       FROM faces
                       WHERE dataset_id = ? AND ground_truth_label != 'None' AND face_id > ?
                       ORDER BY face_id LIMIT ?""",
                    (dataset, last_face_id, batch)
                )
            rows = read_cursor.fetchall()
            if not rows:
                break

            dialog = QDialog(self.ui)
            dialog.setWindowTitle("Usuwanie rekordow")
            dialog.setMinimumSize(900, 700)
            dialog_layout = QVBoxLayout(dialog)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            grid_widget = QWidget()
            inner_grid = QGridLayout(grid_widget)

            check_boxes = {}
            for i, row in enumerate(rows):
                fid, _, face_path = row

                container = QVBoxLayout()
                img_label = QLabel()
                pixmap = QPixmap(face_path)
                if not pixmap.isNull():
                    img_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                chk = QCheckBox("Usun ta osobe")
                chk.setChecked(False)

                container.addWidget(img_label, alignment=Qt.AlignCenter)
                container.addWidget(chk, alignment=Qt.AlignCenter)

                frame = QFrame()
                frame.setStyleSheet("background-color: #3d3d3d; border-radius: 5px;")
                frame.setLayout(container)
                inner_grid.addWidget(frame, i // 4, i % 4)
                check_boxes[fid] = chk

            scroll.setWidget(grid_widget)
            dialog_layout.addWidget(scroll)

            btn_commit_continue = QPushButton("Usun zaznaczone (commit) i dalej")
            btn_commit_continue.setEnabled(True)
            btn_commit_continue.setStyleSheet(
                """
                QPushButton:disabled { background-color: #444; color: #888; }
                QPushButton:enabled { background-color: #28a745; color: white; font-weight: bold; }
                height: 40px;
                """
            )
            btn_commit_stop = QPushButton("Usun zaznaczone (commit) i zakoncz")
            btn_cancel = QPushButton("Anuluj")

            action = {"type": "cancel"}

            def _commit_continue():
                action["type"] = "continue"
                dialog.accept()

            def _commit_stop():
                action["type"] = "stop"
                dialog.accept()

            btn_commit_continue.clicked.connect(_commit_continue)
            btn_commit_stop.clicked.connect(_commit_stop)
            btn_cancel.clicked.connect(dialog.reject)

            dialog_layout.addWidget(btn_commit_continue)
            dialog_layout.addWidget(btn_commit_stop)
            dialog_layout.addWidget(btn_cancel)

            if dialog.exec_() != QDialog.Accepted:
                break

            records_to_be_deleted = [fid for fid, cb in check_boxes.items() if cb.isChecked()]
            for fid in records_to_be_deleted:
                print("Usuwanie rekordu: ", fid, " z datasetu ", dataset)
                self.db._cursor.execute(
                    """DELETE FROM faces WHERE face_id = ? AND dataset_id = ?""",
                    (fid, dataset)
                )
            self.db._conn.commit()

            # Keyset pagination avoids OFFSET shifts after deletes.
            last_face_id = rows[-1][0]

            if action["type"] == "stop":
                break

    def change_record(self, dataset: int = 1):
        old_img_path, ok = QInputDialog.getText(
            None,
            "Podaj sciezke™",
            "Wklej sciezke do pliku z twarza (zamknij okno by zakonczyc):"
        )

        if ok and old_img_path:
            print("Podana sciezke:", old_img_path)

        else:
            print("Anulowano lub brak danych")
            exit(0)

        self.db._cursor.execute("""SELECT face_id FROM faces WHERE dataset_id = ? AND image_path = ?"""
                                , (dataset, old_img_path))

        face_id = self.db._cursor.fetchone()[0]

        new_label, ok = QInputDialog.getText(
            None,
            "Podaj nowa etykiete",
            "Wklej nowa etykiete:"
        )

        if ok and new_label:
            print("Podana etykieta:", new_label)

        else:
            print("Anulowano lub brak danych")

        self._rename_face_and_sync(face_id, new_label)


    def run(self) -> None:

        """Main workflow for clustering and batch labeling."""
        print("[INFO] Start etykietowania ground truth przez DBSCAN + GUI.")

        # 1. Konfiguracja poczatkowa przez GUI
        if not self._setup_session_via_gui():
            print("[INFO] Operacja przerwana przez uzytkownika.")
            return

        print("Dataset domyslny: ", self.dataset_id)
        if self.mode == "align faces":
            self.preprocessor = FacePreprocessor(dataset_id=self.dataset_id, db=self.db, cf=self.config)
            self.preprocessor.compute_embedding_from_crop(alignment = True)

        elif self.mode == "change record":
            while True:
                self.change_record(self.dataset_id)

        elif self.mode == "copy dataset":
            src_dataset = self.ui.ask_for_scan_dataset_id("Wybor zestawu danych", "Wybierz zestaw zrodlowy:")
            print("Dataset zrodlowy: ", src_dataset)
            dst_dataset = self.ui.ask_for_scan_dataset_id("Wybor zestawu danych", "Wybierz zestaw docelowy:")
            print("Dataset docelowy: ", dst_dataset)
            if src_dataset == dst_dataset:
                QMessageBox.warning(
                    self.ui,
                    "Niepoprawny wybor",
                    "Dataset zrodlowy i docelowy musza byc rozne.",
                )
                return
            self.copy_dataset(src_dataset, dst_dataset)
            return

        elif self.mode == "delete record":
            self.delete_records(self.dataset_id)

        elif self.mode == "recalculate face vectors":
            self.preprocessor = FacePreprocessor(dataset_id=self.dataset_id, db=self.db, cf=self.config)
            self.preprocessor.compute_embedding_from_crop()

        else:
            # 2. Pobranie niepodpisanych danych
            fids, embeddings = self._get_unlabeled_data()
            print("Dlugosc wektora twarzy: ", len(embeddings))

            if len(fids) == 0:
                QMessageBox.information(self.ui, "Gotowe", "Wszystkie twarze maja juz etykiety.")
                return

            if len(fids) < self.min_cluster_size:
                QMessageBox.warning(self.ui, "Brak danych", "Za malo niepodpisanych twarzy by uzyc DBSCAN.")
                return

            print(f"[INFO] Rozpoczynam klastrowanie {len(fids)} twarzy...")

            # 3. Uruchomienie DBSCAN (metoda z FaceClassifier)
            clusters_dict = self.classifier.get_face_clusters(embeddings, fids)

            # Odfiltrowanie malych klastrow i sortowanie od najwiekszych
            valid_clusters = [cfids for cid, cfids in clusters_dict.items() if len(cfids) >= self.min_cluster_size]
            print(len(valid_clusters))
            valid_clusters.sort(key=len, reverse=True)

            if not valid_clusters:
                QMessageBox.information(self.ui, "Wynik DBSCAN", "Algorytm nie znalazl zadnych wyraznych grup.")
                return

            # 4. Przekazanie klastrow do GUI do masowego zatwierdzenia
            rounds_without_progress = 0

            to_classify = len(fids)
            for cluster_fids in valid_clusters:
                print("Liczba twarzy: ", to_classify)
                # Pobieramy peĹ‚ne pary (fid, path) by UI mogĹ‚o wyĹ›wietliÄ‡ obrazki
                cluster_data = self.db.get_paths_for_fids(cluster_fids, dataset=self.dataset_id)

                # Wywolanie Bulk UI
                selected_fids, label_name = self.ui.bulk_verify_faces(cluster_data)

                # ObsĹ‚uga przycisku "Anuluj" lub zamkniÄ™cia okna
                if selected_fids is None and label_name is None:
                    print("[INFO] Przerwano etykietowanie na zadanie uzytkownika.")
                    break

                # jesli uzytkownik cos podpisal
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
                        "Brak postepu",
                        "Pominieto kilka grup z rzedu. Przerywam automatyczne podpowiadanie."
                    )
                    break

            print("[INFO] Zakonczono sesje etykietowania.")
            QMessageBox.information(self.ui, "Koniec", "Sesja klastrowania i podpisywania zakonczona.")


if __name__ == "__main__":
    # Inicjalizacja zaleznosci przed startem logiki
    app_config = Config()
    app_db = FaceDatabase(app_config)

    tool = GroundTruthClusterTool(config=app_config, db=app_db)


    try:
        tool.run()
    finally:
        # bezpiecznie zamykamy baze na koncu pliku glownego
        app_db.close()
