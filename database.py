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
        """Standardowa definicja nowej struktury bazy danych."""
        # Tabela faces z nowymi kolumnami
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT,
                dataset_id INTEGER DEFAULT 1,
                original_image TEXT,
                image_path TEXT,
                bbox TEXT,
                embedding BLOB,
                manual_label TEXT,
                svm_prediction TEXT,
                ground_truth_label TEXT DEFAULT NULL,
                is_test INTEGER DEFAULT 0
            )
            """
        )

        # Pozostałe tabele i indeksy
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS processed_images
            (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT , dataset_id INT DEFAULT 1)"""
        )

        # Nowy indeks dla dataset_id (bardzo przyspieszy filtrowanie Twoich 3 zbiorów)
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON faces(dataset_id)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_image ON faces(image_path)")

        self._conn.commit()

    def save_face(self, orig_image, face_img, face_id, bbox, dataset = 1, embedding=None, is_test=0, ground_truth = None
                  , manual_label=None, svm_prediction=None):
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
                INSERT OR REPLACE INTO faces (original_image, face_id, dataset_id,  image_path, bbox, embedding,
                manual_label, svm_prediction, ground_truth_label, is_test)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (orig_image, face_id, dataset, face_path, json.dumps(bbox), emb_json, manual_label
                     , svm_prediction, ground_truth, is_test))
        except Exception as e:
            print(f"[SQL ERROR] save_face: {e}")

    def set_manual_label(self, face_id: str, label: str, dataset: int = 1, is_test: int = 0) -> None:
        """Store a user-provided label for a face entry."""
        self._cursor.execute(
            "UPDATE faces SET manual_label = ?, is_test = ? WHERE face_id = ? AND dataset_id = ?",
            (label, is_test, face_id, dataset)
        )
        self._conn.commit()

    def set_svm_prediction(self, face_id: str, label: str, dataset: int = 1) -> None:
        """Store a model-predicted label for a face entry."""
        self._cursor.execute("UPDATE faces SET svm_prediction = ? WHERE face_id = ? AND dataset_id = ?",
                             (label, face_id, dataset))
        self._conn.commit()

    def mark_as_processed(self, image_path: str, dataset: int = 1) -> None:
        """Mark an original image path as already scanned."""
        self._cursor.execute("INSERT OR IGNORE INTO processed_images (path, dataset_id) VALUES (?, ?)"
                             ,(image_path, dataset))
        self._conn.commit()

    def get_all_processed_paths(self, dataset: int = 1) -> list:
        """Return all source image paths that were already processed."""
        self._cursor.execute("SELECT path FROM processed_images WHERE dataset_id = ?", (dataset, ))
        return [row[0] for row in self._cursor.fetchall()]

    def get_all_unlabeled_embeddings(self, dataset: int = 1):
        self._cursor.execute(
            """SELECT face_id, embedding, image_path FROM faces WHERE
            dataset_id = ? AND ground_truth_label IS NULL""",
            (dataset, )
        )
        rows = self._cursor.fetchall()
        missing_faces = 0
        for fid, emb, p in rows:
            if not os.path.exists(p):
                rows.remove((fid, p))
                print("Brak pliku dla: ", fid, "\tsciezka: ", p, "\n")
                missing_faces += 1
        print(f"Dla {missing_faces} rekordow w bazie danych nie znaleziono pliku z twarzą")
        return [(fid, np.array(json.loads(emb)).astype(float)) for fid, emb, _ in rows]


    def get_all_embeddings_without_ground_truth(self, dataset: int = 1) -> list:
        """Return unlabeled faces as `(face_id, embedding_np)` tuples."""
        self._cursor.execute(
            """SELECT face_id, embedding, image_path FROM faces WHERE ground_truth_label != 'None'
            AND embedding IS NOT NULL AND dataset_id = ? """,
            (dataset,)
        )
        rows = self._cursor.fetchall()
        missing_faces = 0
        for fid, emb, p in rows:
            if not os.path.exists(p):
                rows.remove((fid, p))
                print("Brak pliku dla: ", fid)
                missing_faces += 1
        print(f"Dla {missing_faces} rekordow w bazie danych nie znaleziono pliku z twarzą")
        return [(fid, np.array(json.loads(emb)).astype(float)) for fid, emb, _  in rows]

    def get_unlabeled_test_data(self, dataset: int = 1) -> list:
        """Return unlabeled test records as `(face_id, path, embedding_np, bbox)` tuples."""
        self._cursor.execute(
            """
            SELECT face_id, image_path, embedding, bbox
            FROM faces
            WHERE dataset_id = ?
            AND manual_label IS NULL
            AND embedding IS NOT NULL
            AND ground_truth_label != 'None'
            AND is_test = 1
            """, (dataset, )
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

    def get_labeled_data_for_train(self, dataset: int = 1) -> list:
        """Return training entries with manual labels."""
        self._cursor.execute(
            """
            SELECT face_id, manual_label, embedding FROM faces
            WHERE dataset_id = ? AND manual_label IS NOT NULL AND is_test = 0 AND ground_truth_label != 'None'
            """, (dataset, )
        )
        rows = self._cursor.fetchall()
        return [(fid, label, np.array(json.loads(emb)).astype(float)) for fid, label, emb in rows]

    def get_all_labeled_faces(self, dataset: int = 1) -> list:
        """Return all labeled faces with manual labels preferred over model labels."""
        self._cursor.execute(
            """
            SELECT original_image, face_id,
                   COALESCE(manual_label, svm_prediction) AS final_label,
                   manual_label,
                   image_path,
                   bbox
            FROM faces
            WHERE dataset_id = ? AND manual_label IS NOT NULL OR svm_prediction IS NOT NULL
            """, (dataset, )
        )
        return self._cursor.fetchall()

    def exists_in_db(self, image_path: str, dataset: int = 1) -> bool:
        """Check if a face image path is already in the database."""
        self._cursor.execute("SELECT 1 FROM faces WHERE image_path = ? AND dataset_id = ?"
                             , (image_path, dataset))
        return self._cursor.fetchone() is not None

    def clear_database(self, dataset: int = 1) -> None:
        """Delete all face and processed-image records."""
        self._cursor.execute("DELETE FROM faces WHERE dataset_id = ?", (dataset, ))
        self._cursor.execute("DELETE FROM processed_images WHERE dataset_id = ?", (dataset, ))
        self._conn.commit()
        print("Database cleared.")

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def get_total_faces_count(self, dataset: int = 1) -> int:
        """Return total number of detected faces stored in the database."""
        try:
            self._cursor.execute("SELECT COUNT(*) FROM faces WHERE dataset_id = ?", (dataset,)),
            return self._cursor.fetchone()[0]
        except Exception as e:
            print(f"Error while counting faces: {e}")
            return 0

    def mark_unlabeled_as_test(self, dataset: int = 1):
        """Flag all currently unlabeled entries as test samples."""
        self._cursor.execute("UPDATE faces SET is_test = 1 WHERE manual_label IS NULL AND dataset_id = ?",
                             (dataset, ) )
        self._conn.commit()

    def get_label_by_id(self, face_id, dataset: int = 1) -> str:
        """Return the current label for `face_id`, preferring manual labels."""
        query = "SELECT COALESCE(manual_label, svm_prediction) FROM faces WHERE face_id = ? AND dataset_id = ?"
        self._cursor.execute(query, (face_id, dataset))

        result = self._cursor.fetchone()
        return result[0] if result and result[0] else "Unknown"

    def rename_face_record(self, old_face_id: str, new_face_id: str, new_image_path: str, dataset: int = 1) -> str:
        """Rename record and update path.

        Returns one of: `updated`, `missing_old`, `collision`.
        """
        self._cursor.execute("SELECT 1 FROM faces WHERE face_id = ? AND dataset_id = ?",
                             (old_face_id,dataset, ))
        if not self._cursor.fetchone():
            return "missing_old"

        if old_face_id == new_face_id:
            self._cursor.execute(
                "UPDATE faces SET image_path = ? WHERE face_id = ? AND dataset_id = ?",
                (new_image_path, old_face_id, dataset),
            )
            self._conn.commit()
            return "updated"

        self._cursor.execute("SELECT 1 FROM faces WHERE face_id = ? AND dataset_id = ?",
                             (new_face_id,dataset))
        if self._cursor.fetchone():
            return "collision"

        self._cursor.execute(
            """UPDATE faces SET face_id = ?, image_path = ? , ground_truth_label = ? 
            WHERE face_id = ? AND dataset_id = ?""",
            (new_face_id, new_image_path, self.get_gt_from_path(new_image_path), old_face_id, dataset),
        )
        self._conn.commit()
        return "updated"


    def rebuild_db_from_files(self, dataset: int = 1):
        """Clear only label fields for existing rows, keeping all other metadata intact."""
        print("Clearing DB fields: manual_labels and svm_predictions...")
        self._cursor.execute(
            """
            UPDATE faces
            SET manual_label = NULL,
                svm_prediction = NULL
            WHERE manual_label IS NOT NULL OR svm_prediction IS NOT NULL AND dataset_id = ?
            """, (dataset, )
        )
        updated_rows = self._cursor.rowcount
        self._conn.commit()
        print(f"Labels cleared in {updated_rows} records.")


    @staticmethod
    def get_gt_from_path(path):
        """Extract expected label from filename convention `name_number.ext`."""
        filename = os.path.basename(path)
        name_part = os.path.splitext(filename)[0]
        parts = name_part.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return name_part

    def embedding_generator(self, dataset: int = 1):
        read_cursor = self._conn.cursor()
        read_cursor.execute(
            "SELECT face_id, image_path, embedding FROM faces WHERE dataset_id = ?",# mozna dodac warunek ground_truth == 'None'
            (dataset, )
        )
        missing_files = 0
        while True:
            row = read_cursor.fetchone()
            if row is None:
                break
            face_id, image_path, emb_json = row
            if not os.path.exists(image_path):
                missing_files += 1
                continue
            if emb_json:
                emb = np.array(json.loads(emb_json), dtype=np.float32)
                yield face_id, image_path, emb
        print(f"Missing file for {missing_files} paths")


    def update_emd(self, face_emb, face_id, dataset: int = 1):
        try:
            if face_emb is not None:
                # Zamiana z powrotem na listę i JSON
                emb_list = face_emb.tolist() if isinstance(face_emb, np.ndarray) else face_emb
                emb_json = json.dumps(emb_list)

                self._cursor.execute(
                    "UPDATE faces SET embedding = ? WHERE face_id = ? AND dataset_id = ?",
                    (emb_json, face_id, dataset)
                )
        except Exception as e:
            print(f"[DB ERROR] Failed to compute/save new embedding for {face_id}: {e}")
            return False
        self._conn.commit()
        return True

    def get_paths_for_fids(self, fids: list, dataset: int = 1) -> dict:
        """Zwraca listę ścieżek do cropów dla podanych face_ids."""
        fids_and_paths = {}
        for fid in fids:
            self._cursor.execute("SELECT image_path FROM faces WHERE face_id = ? AND dataset_id = ?"
                                 , (fid,dataset))
            path = self._cursor.fetchone()[0]
            fids_and_paths[fid] = path
        return fids_and_paths
