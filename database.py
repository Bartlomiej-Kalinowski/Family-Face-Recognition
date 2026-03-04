import sqlite3
import os
import cv2
import json
import numpy as np


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
        """Tworzy schemat bazy od zera."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                image_path TEXT,
                bbox TEXT,
                label TEXT,
                embedding BLOB
            )
        ''')
        # Ważne: indeks też musi celować w istniejącą kolumnę!
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_image ON faces(image_path)')
        self.conn.commit()

    def save_face(self, face_img, face_id, original_path, bbox, embedding=None):
        """Zapisuje wycinek twarzy do pliku i metadane do SQL."""
        faces_dir = os.path.abspath(self.config.FACES_DIR)
        os.makedirs(faces_dir, exist_ok=True)
        face_path = os.path.join(faces_dir, f"{face_id}.jpg")

        if not cv2.imwrite(face_path, face_img):
            print(f"[KRYTYCZNY BŁĄD] cv2.imwrite nie powiódł się: {face_path}")
            return

        # Przygotowanie embeddingu do zapisu (konwersja na czysty JSON)
        emb_json = None
        if embedding is not None:
            # Konwersja numpy -> list of floats
            emb_json = json.dumps(embedding.astype(float).tolist())

        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO faces (face_id, image_path, bbox, label, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (face_id, original_path, json.dumps(bbox), None, emb_json))
            self.conn.commit()
        except Exception as e:
            print(f"[BŁĄD SQL] save_face: {e}")

    def update_embedding(self, face_id, embedding):
        """Aktualizuje embedding, obsługując zarówno tablice NumPy, jak i listy."""
        try:
            # 1. Sprawdzamy, czy to tablica NumPy (ma atrybut 'astype')
            if hasattr(embedding, "astype"):
                clean_embedding = embedding.astype(float).tolist()
            # 2. Jeśli to już jest lista, upewniamy się tylko, że elementy to floaty
            elif isinstance(embedding, list):
                clean_embedding = [float(x) for x in embedding]
            # 3. W każdym innym przypadku (np. gdyby to był pojedynczy float lub None)
            else:
                clean_embedding = embedding

            self.cursor.execute('''
                UPDATE faces 
                SET embedding = ? 
                WHERE face_id = ?
            ''', (json.dumps(clean_embedding), face_id))
            self.conn.commit()

        except Exception as e:
            print(f"[BŁĄD SQL] update_embedding dla {face_id}: {e}")

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
        """Pobiera niepodpisane dane i od razu konwertuje je na tablice NumPy."""
        self.cursor.execute('SELECT face_id, embedding FROM faces WHERE label IS NULL AND embedding IS NOT NULL')
        rows = self.cursor.fetchall()

        processed_data = []
        for face_id, emb_json in rows:
            try:
                # Konwersja powrotna: JSON string -> NumPy array
                embedding_array = np.array(json.loads(emb_json), dtype=float)
                processed_data.append((face_id, embedding_array))
            except:
                continue

        return processed_data

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

    def mark_as_processed(self, image_path):
        """Oznacza zdjęcie jako przetworzone w osobnej tabeli."""
        try:
            # Tworzymy tabelę, jeśli jeszcze nie istnieje
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_images (
                    path TEXT PRIMARY KEY
                )
            ''')
            # Wstawiamy ścieżkę (OR IGNORE zapobiega błędom przy duplikatach)
            self.cursor.execute("INSERT OR IGNORE INTO processed_images (path) VALUES (?)", (image_path,))
            self.conn.commit()
        except Exception as e:
            print(f"Błąd bazy danych (mark_as_processed): {e}")

    def get_all_processed_paths(self):
        """Zwraca listę wszystkich przetworzonych ścieżek."""
        try:
            self.cursor.execute("SELECT path FROM processed_images")
            return [row[0] for row in self.cursor.fetchall()]
        except:
            # Jeśli tabela jeszcze nie istnieje, zwracamy pustą listę
            return []