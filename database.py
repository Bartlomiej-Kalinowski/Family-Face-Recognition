"""Database access layer for face metadata and image persistence."""

import json
import os
import sqlite3

import cv2
import numpy as np


class FaceDatabase:
    """Handle SQLite operations and file storage for extracted faces."""

    def __init__(self, config):
        """Initialize storage folders and open the SQLite connection."""
        self.config = config
        self._setup_folders()
        self._conn = sqlite3.connect(
            os.path.join(self.config.OUTPUT_DIR, "face_data.db"),
            check_same_thread=False,
        )
        self._cursor = self._conn.cursor()
        self._create_tables()

    def _setup_folders(self) -> None:
        """Ensure required output folders exist."""
        os.makedirs(self.config.ANNOTATED_FACES_DIR, exist_ok=True)
        os.makedirs(self.config.FACES_DIR, exist_ok=True)

    def _create_tables(self) -> None:
        """Create the database schema when it is missing."""
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                image_path TEXT,
                bbox TEXT,
                embedding BLOB,
                manual_label TEXT,
                svm_prediction TEXT,
                is_test INTEGER DEFAULT 0
            )
            """
        )
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_images (
                path TEXT PRIMARY KEY
            )
            """
        )
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_image ON faces(image_path)")
        self._conn.commit()

    def save_face(self, face_img, face_id, original_path, bbox, embedding=None, is_test=0):
        """Save a cropped face image and its metadata entry."""
        face_path = os.path.join(self.config.FACES_DIR, f"{face_id}.jpg")

        # Persist the crop first so database rows never point to missing files.
        if not cv2.imwrite(face_path, face_img):
            print(f"[I/O ERROR] Cannot save file: {face_path}")
            return

        emb_json = json.dumps(embedding) if embedding is not None else None

        try:
            self._cursor.execute(
                """
                INSERT OR REPLACE INTO faces (face_id, image_path, bbox, manual_label, embedding, is_test)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (face_id, original_path, json.dumps(bbox), None, emb_json, is_test),
            )
        except Exception as e:
            print(f"[SQL ERROR] save_face: {e}")

    def set_manual_label(self, face_id: str, label: str, is_test: int = 0) -> None:
        """Store a user-provided label for a face entry."""
        self._cursor.execute(
            "UPDATE faces SET manual_label = ?, is_test = ? WHERE face_id = ?",
            (label, is_test, face_id),
        )
        self._conn.commit()

    def set_svm_prediction(self, face_id: str, label: str) -> None:
        """Store a model-predicted label for a face entry."""
        self._cursor.execute("UPDATE faces SET svm_prediction = ? WHERE face_id = ?", (label, face_id))
        self._conn.commit()

    def mark_as_processed(self, image_path: str) -> None:
        """Mark an original image path as already scanned."""
        self._cursor.execute("INSERT OR IGNORE INTO processed_images (path) VALUES (?)", (image_path,))
        self._conn.commit()

    def get_all_processed_paths(self) -> list:
        """Return all source image paths that were already processed."""
        self._cursor.execute("SELECT path FROM processed_images")
        return [row[0] for row in self._cursor.fetchall()]

    def get_all_unlabeled_embeddings(self) -> list:
        """Return unlabeled faces as `(face_id, embedding_np)` tuples."""
        self._cursor.execute(
            "SELECT face_id, embedding FROM faces WHERE manual_label IS NULL AND embedding IS NOT NULL"
        )
        rows = self._cursor.fetchall()
        return [(fid, np.array(json.loads(emb)).astype(float)) for fid, emb in rows]

    def get_unlabeled_test_data(self) -> list:
        """Return unlabeled test records as `(face_id, path, embedding_np, bbox)` tuples."""
        self._cursor.execute(
            """
            SELECT face_id, image_path, embedding, bbox
            FROM faces
            WHERE manual_label IS NULL
            AND embedding IS NOT NULL
            AND is_test = 1
            """
        )
        rows = self._cursor.fetchall()

        processed_data = []
        for fid, path, emb_json, bbox in rows:
            try:
                emb_np = np.array(json.loads(emb_json)).astype(float)
                processed_data.append((fid, path, emb_np, bbox))
            except Exception as e:
                print(f"[ERROR] Failed to parse embedding for {fid}: {e}")

        return processed_data

    def get_labeled_data_for_train(self) -> list:
        """Return training entries with manual labels."""
        self._cursor.execute(
            """
            SELECT face_id, manual_label, embedding FROM faces
            WHERE manual_label IS NOT NULL AND is_test = 0
            """
        )
        rows = self._cursor.fetchall()
        return [(fid, label, np.array(json.loads(emb)).astype(float)) for fid, label, emb in rows]

    def get_all_labeled_faces(self) -> list:
        """Return all labeled faces with manual labels preferred over model labels."""
        self._cursor.execute(
            """
            SELECT face_id,
                   COALESCE(manual_label, svm_prediction) AS final_label,
                   manual_label,
                   image_path
            FROM faces
            WHERE manual_label IS NOT NULL OR svm_prediction IS NOT NULL
            """
        )
        return self._cursor.fetchall()

    def clear_database(self) -> None:
        """Delete all face and processed-image records."""
        self._cursor.execute("DELETE FROM faces")
        self._cursor.execute("DELETE FROM processed_images")
        self._conn.commit()
        print("Database cleared.")

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def get_total_faces_count(self) -> int:
        """Return total number of detected faces stored in the database."""
        try:
            self._cursor.execute("SELECT COUNT(*) FROM faces")
            return self._cursor.fetchone()[0]
        except Exception as e:
            print(f"Error while counting faces: {e}")
            return 0

    def mark_unlabeled_as_test(self):
        """Flag all currently unlabeled entries as test samples."""
        self._cursor.execute("UPDATE faces SET is_test = 1 WHERE manual_label IS NULL")
        self._conn.commit()

    def get_label_by_id(self, face_id) -> str:
        """Return the current label for `face_id`, preferring manual labels."""
        query = "SELECT COALESCE(manual_label, svm_prediction) FROM faces WHERE face_id = ?"
        self._cursor.execute(query, (face_id,))

        result = self._cursor.fetchone()
        return result[0] if result and result[0] else "Unknown"
