"""GUI tool for DBSCAN-assisted ground-truth labeling of extracted faces."""

from __future__ import annotations

from collections import deque
import os
import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from sklearn.cluster import DBSCAN

from config import Config
from database import FaceDatabase
from interface import FaceInterface
from rename_tool import _rename_file_and_sync_db

from ml_engine import FaceExtractor



class GroundTruthClusterTool:
    """Label all unlabeled faces by iterating DBSCAN clusters and manual corrections."""

    def __init__(self, min_cluster_size: int = 3, max_cluster_preview: int = 24):
        self.config = Config()
        self.db = FaceDatabase(self.config)
        self.ui = FaceInterface()
        self.min_cluster_size = max(2, int(min_cluster_size))
        # self.max_cluster_preview = max(4, int(max_cluster_preview))
        self._label_next_index: dict[str, int] = {}

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Keep filename-safe characters for generated face IDs."""
        return re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")

    def _ensure_label_counter(self, safe_label: str) -> None:
        """Initialize next index for label based on existing rows in DB."""
        if safe_label in self._label_next_index:
            return

        self.db._cursor.execute(
            "SELECT face_id FROM faces WHERE face_id LIKE ?",
            (f"{safe_label}_%",),
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

            self.db._cursor.execute("SELECT 1 FROM faces WHERE face_id = ?", (candidate,))
            db_has_id = self.db._cursor.fetchone() is not None
            fs_has_file = os.path.exists(candidate_path)
            if not db_has_id and not fs_has_file:
                self._label_next_index[safe_label] = next_idx + 1
                return candidate, os.path.abspath(candidate_path)

            next_idx += 1

    def _rename_face(self, old_face_id: str, safe_label: str) -> str:
        """Rename crop file and synchronize `face_id`/`image_path` via DB API."""
        new_face_id, new_face_path = self._allocate_new_name(safe_label)
        old_face_path = os.path.join(self.config.FACES_DIR, f"{old_face_id}.jpg")
        _rename_file_and_sync_db(self.db,
                                 old_face_path,
                                 new_face_path,
                                 old_face_id,
                                 new_face_id )

    def _get_unlabeled_map(self) -> dict[str, np.ndarray]:
        """Return `{face_id: embedding}` for all unlabeled rows."""
        rows = self.db.get_all_embeddings_without_ground_truth()
        return {fid: emb for fid, emb in rows}

    def _cluster(self, pending_ids: list[str], emb_map: dict[str, np.ndarray]) -> list[list[str]]:
        """Run DBSCAN on pending IDs and return valid clusters sorted by size."""
        if len(pending_ids) < self.min_cluster_size:
            return []

        embeddings = np.array([emb_map[fid] for fid in pending_ids], dtype=np.float32)
        labels = DBSCAN(eps=0.25, min_samples=2, metric="cosine").fit_predict(embeddings)


        raw: dict[int, list[str]] = {}
        for fid, label in zip(pending_ids, labels):
            raw.setdefault(int(label), []).append(fid)

        valid = []
        for cid, fids in raw.items():
            if cid == -1:
                continue
            if len(fids) >= self.min_cluster_size:
                valid.append(fids)

        valid.sort(key=len, reverse=True)
        return valid

    def _assign_label(self, face_ids: list[str], label: str) -> int:
        """Rename faces and persist ground-truth label for selected records."""
        clean_label = label.strip()
        safe_label = self._sanitize_label(clean_label)
        if not clean_label or not safe_label:
            return 0

        written = 0
        for idx, fid in enumerate(face_ids, start=1):
            self._rename_face(fid, safe_label)
            # Keep manual label for training/query compatibility while IDs follow rename_tool style.
            written += 1
            self.ui.update_progress(idx, len(face_ids), f"Zapisywanie: {clean_label}")
            QApplication.processEvents()
        return written

    def _review_cluster(self, cluster_ids: list[str]) -> tuple[bool, int]:
        """Show cluster dialog and save selected faces.

        Returns `(continue_labeling, written_count)`.
        """
        preview_ids = cluster_ids
        selected, name = self.ui.bulk_verify_faces(preview_ids)
        if selected is None and name is None:
            return False, 0
        if not selected:
            return True, 0
        return True, self._assign_label(selected, name)

    def _review_single_face(self, face_id: str) -> tuple[bool, int]:
        """Fallback labeling for outliers that do not form DBSCAN clusters."""
        selected, name = self.ui.bulk_verify_faces([face_id])
        if selected is None and name is None:
            return False, 0
        if not selected:
            return True, 0
        return True, self._assign_label(selected, name)

    def run(self) -> None:
        """Drive the full loop until all faces are labeled or user exits."""
        print("[INFO] Start etykietowania ground truth przez DBSCAN + GUI.")
        rounds_without_progress = 0

        while True:
            emb_map = self._get_unlabeled_map() # all embeddings without ground truth label in DB
            pending = list(emb_map.keys())
            if not pending:
                print("[SUKCES] Wszystkie twarze zostały podpisane.")
                QMessageBox.information(self.ui, "Gotowe", "Wszystkie twarze mają etykiety ground truth.")
                return

            print(f"[INFO] Pozostało niepodpisanych twarzy: {len(pending)}")
            clusters = self._cluster(pending, emb_map)
            progress_in_round = 0

            if clusters:
                keep_going, written = self._review_cluster(clusters[0])
                if not keep_going:
                    print("[INFO] Przerwano na żądanie użytkownika.")
                    return
                progress_in_round += written
            else:
                queue = deque(pending)
                while queue and progress_in_round == 0:
                    fid = queue.popleft()
                    keep_going, written = self._review_single_face(fid)
                    if not keep_going:
                        print("[INFO] Przerwano na żądanie użytkownika.")
                        return
                    progress_in_round += written
                    if written == 0:
                        queue.append(fid)
                    if len(queue) == len(pending) and progress_in_round == 0:
                        break

            if progress_in_round == 0:
                rounds_without_progress += 1
                if rounds_without_progress >= 2:
                    QMessageBox.warning(
                        self.ui,
                        "Brak postępu",
                        "Nie zapisano żadnej etykiety w ostatnich iteracjach.\n"
                        "Odkliknięte twarze będą pojawiać się ponownie, ale bez potwierdzeń "
                        "narzędzie nie zakończy podpisywania wszystkich twarzy.",
                    )
                    return
            else:
                rounds_without_progress = 0


if __name__ == "__main__":
    tool = GroundTruthClusterTool()
    is_changed_embds = int(input("recompute embds?(0/1): 0 - no, 1 - yes:\t"))
    if is_changed_embds == 1:
        tool.db.recompute_all_embeddings(FaceExtractor.compute_embedding_from_crop)
        tool.db.close()
    try:
        tool.run()
    finally:
        tool.db.close()
