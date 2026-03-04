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

    def save_face(self, face_img, face_id, original_path, bbox):
        """Zapisuje wycinek twarzy do pliku i metadane do SQL."""

        # 1. Wymuszamy ścieżkę absolutną i normalizujemy ukośniki
        faces_dir = os.path.abspath(self.config.FACES_DIR)
        os.makedirs(faces_dir, exist_ok=True)

        # 2. Tworzymy bezpieczną ścieżkę pliku
        face_filename = f"{face_id}.jpg"
        face_path = os.path.join(faces_dir, face_filename)
        # 3. Zapis fizyczny
        success = cv2.imwrite(face_path, face_img)
        if not success:
            # Jeśli nadal nie działa, sprawdź czy face_img nie jest None
            print(f"[KRYTYCZNY BŁĄD] cv2.imwrite zwrócił False dla: {face_path}")

    def update_embedding(self, face_id, embedding):
        """Aktualizuje embedding dla konkretnej twarzy."""
        # Konwersja na listę, jeśli to obiekt numpy (dla json.dumps)
        embedding_data = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        # Zmieniamy 'embedding_json' na 'embedding'
        self.cursor.execute('UPDATE faces SET embedding = ? WHERE face_id = ?',
                            (json.dumps(embedding_data), face_id))
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
        """Pobiera ID i embeddingi dla twarzy, które nie mają jeszcze etykiety."""
        # Zmieniamy 'embedding_json' na 'embedding'
        self.cursor.execute('SELECT face_id, embedding FROM faces WHERE label IS NULL')
        return self.cursor.fetchall()

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