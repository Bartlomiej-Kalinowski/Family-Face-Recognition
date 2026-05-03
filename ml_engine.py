"""Machine-learning components for face extraction, clustering, and classification."""
import math
import sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from database import FaceDatabase

try:
    import torch
    from ultralytics import YOLO

    _ = torch.empty(1)
except Exception:
    print("Blad importu biblioteki ultralytics lub pytorch")
    exit(0)

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import BallTree


class FacePreprocessor:
    def __init__(self, dataset_id: int, db: FaceDatabase, cf):
        base_options = mp_python.BaseOptions(model_asset_path=cf.FACE_LANDMARKER_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.recognizer = None
        self.dataset = dataset_id
        self.db = db
        self.config = cf

    def recompute_one_embedding_with_face_alignment(self, image_path: str):
        """
        Wczytuje cropa, wyrównuje linię oczu i zwraca nowy embedding.
        Zwraca None, jeśli wyrównanie się nie powiedzie.
        """

        img = cv2.imread(image_path)
        if img is None:
            print("Brak sciezki do pliku z twarza!")
            return None

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Znalezienie landmarków na cropie
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("Nie znaleziono landamrkow na twarzy!")
            return None

        landmarks = result.face_landmarks[0]

        if len(landmarks) > 473:
            left_idx, right_idx = 468, 473
        else:
            left_idx, right_idx = 33, 263  # kąciki oczu

        left_eye_x = int(landmarks[left_idx].x * w)
        left_eye_y = int(landmarks[left_idx].y * h)
        right_eye_x = int(landmarks[right_idx].x * w)
        right_eye_y = int(landmarks[right_idx].y * h)

        # 3. Obliczenie kąta nachylenia linii oczu
        dy = right_eye_y - left_eye_y
        dx = right_eye_x - left_eye_x
        angle = math.degrees(math.atan2(dy, dx))

        # 4. Obrót obrazu względem środka między oczami
        eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

        aligned_img = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE  # Klonuje krawędzie, by nie było czarnych dziur
        )

        # 5. Skalowanie do SFace (112x112) i wyciągnięcie embeddingu
        aligned_resized = cv2.resize(aligned_img, (112, 112))
        embedding = self.recognizer.feature(aligned_resized)

        return embedding.flatten().astype(np.float32)

    def close(self):
        self.face_landmarker.close()

    @staticmethod
    def recompute_one_embedding(face_image_path):
        # Wczytujemy fizyczny plik obrazu z dysku
        img = cv2.imread(face_image_path)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
        img = clahe.apply(gray)

        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        img = hog.compute(img).flatten().astype(np.float32)
        return img


    def compute_embedding_from_crop(self, type_of_preprocessing):
        nb_features = None
        # WERSJA 1: SIEC NEURONOWA
        if type_of_preprocessing == "neural_network":
            self.recognizer = cv2.FaceRecognizerSF.create(
                self.config.FACE_RECOGNIZER_CV_PATH,
                ""
            )
            self.db.clear_embeddings(self.dataset)
            updated_correctly = 0
            for face_id, img_path, face_emb in tqdm(self.db.embedding_generator(self.dataset), "Ekstrakcja cech"):
                print("Before: ", len(face_emb)) if face_emb is not None else None
                face_emb = self.recompute_one_embedding_with_face_alignment(img_path)
                print("After: ", len(face_emb))if face_emb is not None else None
                if face_emb is not None:
                    # Normalizacja L2 (SFace działa najlepiej na sferze)
                    face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-7)

                    if self.db.update_emd(face_emb, face_id, dataset=self.dataset):
                        updated_correctly += 1
                else:
                    print(f"[SKIP] Nie udalo sie obliczyc wektora dla: {face_id}. Pomijam aktualizacje i oznaczam jako None.")
                    self.db.mark_as_none(face_id, dataset=self.dataset)
                    continue
            self.db._conn.commit()
            print(f"\n[DONE] Neural Network update complete. Updated: {updated_correctly}")
            return  # PCA jest zbędne dla SFace

        #WERSJA 2: hog + PCA
        pca = IncrementalPCA(n_components=150)
        scaler = Normalizer(norm='l2')
        updated_correctly = 0
        update_errors = 0
        print("--- Preprocessing, part I: Fitting PCA ---", file=sys.stderr, flush=True)
        batch_size = 150
        batch_embs = []
        self.db.clear_embeddings(self.dataset)
        for face_id, img_path, face_emb in tqdm(self.db.embedding_generator(self.dataset), "Fitting PCA"):
            face_emb = self.recompute_one_embedding(img_path)
            if face_emb is None:
                print(f"[SKIP] Nie wykryto twarzy dla ID: {face_id}. Pomijam aktualizacje.")
                continue  # Nie dodajemy do batcha, nie aktualizujemy bazy, gdy detector nic nie wykryl
            nb_features = face_emb.shape
            # KROK 1: Zapisuje surowy HOG
            success = self.db.update_emd(face_emb, face_id, dataset=self.dataset)
            if success:
                updated_correctly += 1
                batch_embs.append(face_emb)
            else:
                update_errors += 1

            # Jeśli paczka ma 150 elementów, uczymy model i czyścimy paczkę
            if len(batch_embs) == batch_size:
                pca.partial_fit(np.array(batch_embs))
                batch_embs.clear()

        # Dobicie resztki danych, które nie wypełniły pełnej setki
        if len(batch_embs) > 0:
            pca.partial_fit(np.array(batch_embs))


        print(f"Errors, part I: {update_errors}")
        print("Emb shape part I: ", nb_features)
        print(f"\nUpdated correctly, part I: {updated_correctly}")

        print("--- Preprocessing, part II: Transforming & Normalizing ---", file=sys.stderr, flush=True)
        updated_correctly = 0
        update_errors = 0
        nb_features = None

        # 2. Transformacja i normalizacja każdego embeddingu
        for face_id, _, face_emb in tqdm(self.db.embedding_generator(self.dataset), "Transforming"):
            # PCA wymaga danych w formacie 2D, więc robimy reshape(1, -1)
            reduced = pca.transform(face_emb.reshape(1, -1))

            # Skalowanie L2 i powrót do 1D za pomocą flatten()
            final_emb = scaler.transform(reduced).flatten()
            nb_features = final_emb.shape

            # Aktualizacja w bazie - zwraca True/False
            success = self.db.update_emd(final_emb, face_id, dataset=self.dataset)
            if success:
                updated_correctly += 1
            else:
                update_errors += 1

        self.db._conn.commit()

        print("\n[DB] Recalculation complete.")
        print(f"Updated correctly, part II: {updated_correctly}")
        print(f"Errors, part II: {update_errors}")
        print("Emb shape part II: ", nb_features)

class FaceExtractor:
    """Detect faces with YOLO and compute HOG embeddings for each crop."""

    def __init__(self, config, db: FaceDatabase, dataset_id: int):
        self.config = config
        self.detector = YOLO(config.YOLO_MODEL_PATH)
        self.db = db
        self.dataset = dataset_id


    def extract_face_data(self, image_path: str) -> list:
        """Return detected faces as dictionaries with crop, bbox, and embedding."""
        img = cv2.imread(image_path)
        if img is None:
            return []

        results = self.detector(image_path, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        faces_data = []
        img_h, img_w, _ = img.shape


        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Add context around the face to make embeddings less brittle.
            w_box, h_box = x2 - x1, y2 - y1
            pad = 0.0
            x1, y1 = max(0, int(x1 - w_box * pad)), max(0, int(y1 - h_box * pad))
            x2, y2 = min(img_w, int(x2 + w_box * pad)), min(img_h, int(y2 + h_box * pad))

            base_embedding = img[y1:y2, x1:x2]

            if base_embedding is not None:
                faces_data.append({
                    "original_image": image_path,
                    "crop": base_embedding,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "embedding": base_embedding.tolist(),
                })
        return faces_data



class FaceClassifier:
    """Cluster unlabeled faces and run multi-class SVM classification."""

    def __init__(self):
        """Initialize classifier state."""
        self.model = None
        self.is_trained = False

    def get_face_clusters(self, embeddings: np.ndarray, fids: list) -> dict:
        """Group unlabeled embeddings using DBSCAN with cosine distance."""

        dbscan = DBSCAN(eps=0.23, min_samples=1, metric="cosine")

        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for fid, label in zip(fids, labels):
            if label == -1:  # noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fid)
        return clusters


class SVMClassifier(FaceClassifier):
    """SVM classifier for face classification."""
    def train_one_vs_rest_svm(self, x_train: list, y_train_labels: list) -> None:
        """Train an One-vs-Rest SVM pipeline with grid search."""
        if len(set(y_train_labels)) < 2:
            return

        from scipy.stats import loguniform
        from sklearn.model_selection import RandomizedSearchCV
        pipe = Pipeline([
                ("clf", OneVsRestClassifier(
                SVC(kernel="rbf", probability=True, class_weight="balanced"),
                n_jobs=-1
            )),
        ])

        param_grid = {
            "clf__estimator__C": loguniform(1e3, 1e5),
            "clf__estimator__gamma": loguniform(1e-4, 1e-1),
        }

        search = RandomizedSearchCV(
            pipe,
            param_grid,
            n_iter=10,
            n_jobs=-1
        )

        search.fit(np.array(x_train), y_train_labels)

        self.model = search.best_estimator_
        self.is_trained = True

        print(f"Best parameters (OvR): {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.2%}")

    def predict_unlabeled(self, x_test: np.ndarray) -> tuple:
        """Predict labels and confidence scores for unlabeled embeddings."""
        if not self.is_trained:
            return [], []

        predictions = self.model.predict(x_test)
        probabilities = np.max(self.model.predict_proba(x_test), axis=1)

        return predictions, probabilities

class KNNclassifier(FaceClassifier):
    """KNN classifier for face classification."""

    def __init__(self, x_train: np.ndarray, y_train: list[str], distance_threshold: float = 0.6):
        """
        x_train: Macierz embeddingów wygenerowana i zatwierdzona z DBSCAN.
        y_train: Lista etykiet odpowiadająca wierszom w x_train (np. ["Jan", "Jan", "Anna", ...]).
        distance_threshold: Maksymalna odległość euklidesowa. Powyżej tego progu twarz to "Nieznany".
        """
        self.y_train = np.array(y_train)
        self.threshold = distance_threshold
        self.k = 3

        # Budujemy drzewo tylko raz na zatwierdzonych danych
        print("[INFO] Budowanie BallTree...")
        self.kdt = BallTree(x_train, leaf_size=30, metric='euclidean')

    def predict_unlabeled(self, x_test: np.ndarray) -> list[str]:
        """
        Klasyfikuje nowe wektory twarzy na podstawie k najbliższych sąsiadów.
        """
        from collections import Counter
        # Szukamy k-sąsiadów.
        distances, indices = self.kdt.query(x_test, k=self.k, return_distance=True)
        predictions = []

        for i in range(len(x_test)):
            dist_i = distances[i]
            ind_i = indices[i]
            valid_neighbors_labels = []

            for j in range(self.k):
                current_dist = dist_i[j]
                current_idx = ind_i[j]
                if current_dist <= self.threshold:
                    valid_neighbors_labels.append(self.y_train[current_idx])

            if len(valid_neighbors_labels) == 0:
                predictions.append("Nieznana osoba")
            else:
                # Głosowanie większościowe tylko wśród "bliskich" sąsiadów
                counts = Counter(valid_neighbors_labels)
                most_common_label = counts.most_common(1)[0][0]
                predictions.append(most_common_label)

        return predictions, None





