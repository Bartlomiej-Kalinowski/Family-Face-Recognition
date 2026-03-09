import sqlite3
import os
import cv2
import json
import numpy as np


class FaceDatabase:
    """
    Warstwa dostępu do danych (DAO).
    Zarządza bazą SQLite oraz plikami obrazów twarzy.
    """

    def __init__(self, config):
        self.config = config
        self._setup_folders()
        self._conn = sqlite3.connect(os.path.join(self.config.OUTPUT_DIR, "face_data.db"), check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._create_tables()
        self._migrate_db()  # Automatycznie naprawia brakujące kolumny

    def _setup_folders(self) -> None:
        os.makedirs(self.config.ANNOTATED_DIR, exist_ok=True)
        os.makedirs(self.config.FACES_DIR, exist_ok=True)

    def _create_tables(self) -> None:
        """Tworzy bazę w docelowym formacie."""
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                image_path TEXT,
                bbox TEXT,
                embedding BLOB,
                manual_label TEXT,       -- Etykieta od użytkownika
                svm_prediction TEXT,     -- Etykieta od modelu
                is_test INTEGER DEFAULT 0 -- Podział na zbiory
            )
        ''')
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_images (
                path TEXT PRIMARY KEY
            )
        ''')
        self._cursor.execute('CREATE INDEX IF NOT EXISTS idx_image ON faces(image_path)')
        self._conn.commit()

    def _migrate_db(self):
        """Dodaje brakujące kolumny do starej bazy bez jej usuwania."""
        columns_to_add = [
            ("manual_label", "TEXT"),
            ("svm_prediction", "TEXT"),
            ("is_test", "INTEGER DEFAULT 0")
        ]
        for col_name, col_type in columns_to_add:
            try:
                self._cursor.execute(f"ALTER TABLE faces ADD COLUMN {col_name} {col_type}")
                print(f"[MIGRACJA] Dodano kolumnę {col_name}")
            except sqlite3.OperationalError:
                pass  # Kolumna już istnieje
        self._conn.commit()

    # --- ZAPISYWANIE DANYCH ---

    def save_face(self, face_img, face_id, original_path, bbox, embedding=None):
        """Zapisuje wycinek twarzy i metadane."""
        face_path = os.path.join(self.config.FACES_DIR, f"{face_id}.jpg")

        # Zapis fizyczny pliku
        if not cv2.imwrite(face_path, face_img):
            print(f"[BŁĄD I/O] Nie można zapisać: {face_path}")
            return

        # Przygotowanie embeddingu (do JSON)
        emb_json = json.dumps(embedding) if embedding is not None else None

        try:
            self._cursor.execute('''
                INSERT OR REPLACE INTO faces (face_id, image_path, bbox, manual_label, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (face_id, original_path, json.dumps(bbox), None, emb_json))
            self._conn.commit()
        except Exception as e:
            print(f"[BŁĄD SQL] save_face: {e}")

    def set_manual_label(self, face_id: str, label: str, is_test: int = 0) -> None:
        """Zapisuje etykietę użytkownika (Ground Truth)."""
        self._cursor.execute('''
            UPDATE faces SET manual_label = ?, is_test = ? WHERE face_id = ?
        ''', (label, is_test, face_id))
        self._conn.commit()

    def set_svm_prediction(self, face_id: str, label: str) -> None:
        """Zapisuje wynik predykcji modelu."""
        self._cursor.execute('UPDATE faces SET svm_prediction = ? WHERE face_id = ?', (label, face_id))
        self._conn.commit()

    def mark_as_processed(self, image_path: str) -> None:
        self._cursor.execute("INSERT OR IGNORE INTO processed_images (path) VALUES (?)", (image_path,))
        self._conn.commit()

    # --- POBIERANIE DANYCH (GETTERY) ---

    def get_all_processed_paths(self) -> list:
        self._cursor.execute("SELECT path FROM processed_images")
        return [row[0] for row in self._cursor.fetchall()]

    def get_all_unlabeled_embeddings(self) -> list:
        """Pobiera dane, które nie mają jeszcze manual_label (do klastrowania/predykcji)."""
        self._cursor.execute(
            'SELECT face_id, embedding FROM faces WHERE manual_label IS NULL AND embedding IS NOT NULL')
        rows = self._cursor.fetchall()
        return [(fid, np.array(json.loads(emb)).astype(float)) for fid, emb in rows]

    def get_labeled_data_for_train(self) -> list:
        """Pobiera dane oznaczone jako treningowe."""
        self._cursor.execute('''
            SELECT face_id, manual_label, embedding FROM faces 
            WHERE manual_label IS NOT NULL AND is_test = 0
        ''')
        rows = self._cursor.fetchall()
        return [(fid, label, np.array(json.loads(emb)).astype(float)) for fid, label, emb in rows]

    def get_all_labeled_faces(self) -> list:
        """Zwraca pary (id, etykieta) - priorytet ma manual_label, potem svm_prediction."""
        self._cursor.execute(
            "SELECT face_id, COALESCE(manual_label, svm_prediction) FROM faces WHERE manual_label IS NOT NULL OR svm_prediction IS NOT NULL")
        return self._cursor.fetchall()

    def clear_database(self) -> None:
        self._cursor.execute("DELETE FROM faces")
        self._cursor.execute("DELETE FROM processed_images")
        self._conn.commit()
        print("Baza wyczyszczona.")

    def close(self) -> None:
        self._conn.close()

    def get_total_faces_count(self) -> int:
        """Zwraca łączną liczbę wykrytych twarzy w bazie."""
        try:
            self._cursor.execute("SELECT COUNT(*) FROM faces")
            return self._cursor.fetchone()[0]
        except Exception as e:
            print(f"Błąd podczas liczenia twarzy: {e}")
            return 0