import sqlite3
import os
import cv2
import json


class FaceDatabase:
    """Professional Database using SQLite instead of JSON."""

    def __init__(self, config):
        self.config = config
        self._setup_folders()
        # Łączymy się z bazą (plik zostanie utworzony, jeśli nie istnieje)
        self.conn = sqlite3.connect(os.path.join(self.config.OUTPUT_DIR, "face_data.db"), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _setup_folders(self):
        os.makedirs(self.config.ANNOTATED_DIR, exist_ok=True)
        os.makedirs(self.config.FACES_DIR, exist_ok=True)

    def _create_tables(self):
        """Creates the database schema."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                original_image TEXT,
                bbox_json TEXT,
                label TEXT,
                embedding_json TEXT,
                is_validated INTEGER DEFAULT 0
            )
        ''')
        # Indeks dla szybkości wyszukiwania po obrazku
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_image ON faces(original_image)')
        self.conn.commit()

    def save_face(self, face_img, face_id, original_path, bbox):
        """Saves face crop and metadata directly to SQL."""
        face_path = os.path.join(self.config.FACES_DIR, f"{face_id}.jpg")
        cv2.imwrite(face_path, face_img)

        self.cursor.execute('''
            INSERT OR REPLACE INTO faces (face_id, original_image, bbox_json, label)
            VALUES (?, ?, ?, ?)
        ''', (face_id, original_path, json.dumps(bbox), None))
        self.conn.commit()

    def update_embedding(self, face_id, embedding):
        """Updates only the embedding for a specific face."""
        self.cursor.execute('UPDATE faces SET embedding_json = ? WHERE face_id = ?',
                            (json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding), face_id))
        self.conn.commit()

    def get_unlabeled_map(self):
        """Returns the unlabeled faces grouped by image path."""
        self.cursor.execute('SELECT face_id, original_image, bbox_json FROM faces WHERE label IS NULL')
        rows = self.cursor.fetchall()

        unlabeled = {}
        for fid, img_path, bbox_json in rows:
            if img_path not in unlabeled:
                unlabeled[img_path] = []
            unlabeled[img_path].append(fid)
        return unlabeled

    def get_metadata_for_gui(self, face_id):
        """Fetch single face details."""
        self.cursor.execute('SELECT bbox_json, label, embedding_json FROM faces WHERE face_id = ?', (face_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "bbox": json.loads(row[0]),
                "label": row[1],
                "embedding": json.loads(row[2]) if row[2] else None
            }
        return None

    def set_label(self, face_id, label, validated=True):
        """Updates the label for a face."""
        self.cursor.execute('UPDATE faces SET label = ?, is_validated = ? WHERE face_id = ?',
                            (label, 1 if validated else 0, face_id))
        self.conn.commit()

    def save_db(self):
        """In SQL, data is usually saved on commit(), but we keep this for compatibility."""
        self.conn.commit()

    def close(self):
        self.conn.close()

    def get_all_processed_paths(self):
        """Returns a set of all unique original_image paths in the DB."""
        self.cursor.execute('SELECT DISTINCT original_image FROM faces')
        return {row[0] for row in self.cursor.fetchall()}

    def get_all_unlabeled_embeddings(self):
        """Returns list of (face_id, embedding) for unlabeled faces."""
        self.cursor.execute('SELECT face_id, embedding_json FROM faces WHERE label IS NULL')
        rows = self.cursor.fetchall()
        return [(row[0], json.loads(row[1])) for row in rows if row[1]]

    def get_all_labeled_faces(self):
        """Pobiera listę (face_id, label) dla wszystkich podpisanych twarzy."""
        query = "SELECT face_id, label FROM faces WHERE label IS NOT NULL"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def clear_database(self):
        """Usuwa wszystkie rekordy z bazy danych, przygotowując ją na nowy skan."""
        try:
            self.cursor.execute("DELETE FROM faces")
            self.conn.commit()
            print("Baza danych została wyczyszczona.")
        except Exception as e:
            print(f"Błąd podczas czyszczenia bazy: {e}")