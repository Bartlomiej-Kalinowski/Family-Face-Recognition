"""Machine-learning components for face extraction, clustering, and classification."""

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
from sklearn.decomposition import PCA, IncrementalPCA



class FaceExtractor:
    """Detect faces with YOLO and compute HOG embeddings for each crop."""

    def __init__(self, config, db: FaceDatabase, dataset_id: int = 1):
        self.config = config
        self.detector = YOLO(config.YOLO_MODEL_PATH)
        self.db = db
        self.dataset = dataset_id

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

    def compute_embedding_from_crop(self):
        pca = IncrementalPCA(n_components=150)
        scaler = Normalizer(norm='l2')

        print("--- Preprocessing, part I: Fitting PCA ---", flush=True)
        batch_size = 150
        batch_embs = []
        updated_correctly = 0
        update_errors = 0
        nb_features = None
        # 1. Trenowanie PCA paczkami (partial_fit)
        for face_id, img_path, face_emb in tqdm(self.db.embedding_generator(self.dataset), "Fitting PCA"):
            face_emb = self.recompute_one_embedding(img_path)
            nb_features = face_emb.shape
            success = self.db.update_emd(face_emb, face_id, dataset=self.dataset)
            if success:
                updated_correctly += 1
            else:
                update_errors += 1
            batch_embs.append(face_emb)

            # Jeśli paczka ma 150 elementów, uczymy model i czyścimy paczkę
            if len(batch_embs) == batch_size:
                pca.partial_fit(np.array(batch_embs))
                batch_embs.clear()

        # Dobicie resztki danych, które nie wypełniły pełnej setki
        if len(batch_embs) > 0:
            pca.partial_fit(np.array(batch_embs))

        print("--- Preprocessing, part I: Transforming & Normalizing ---", flush=True)
        print(f"Updated correctly, part I: {updated_correctly}")
        print(f"Errors, part I: {update_errors}")
        print("Emb shape part I: ", nb_features)
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

class FaceClusterer:
    def __init__(self, dataset_id: int = 1):
        self.X_normalized = None
        self.dataset = dataset_id

    def get_data(self):
        self.X_normalized = self.db.get_all_embeddings_without_ground_truth(self.dataset)

    def check_data_density(self):
        from sklearn.neighbors import NearestNeighbors
        import matplotlib.pyplot as plt

        # n_neighbors powinno odpowiadać parametrowi min_samples w DBSCAN (zazwyczaj 5)
        neigh = NearestNeighbors(n_neighbors=5)
        nbrs = neigh.fit(self.X_normalized)
        distances, indices = nbrs.kneighbors(self.X_normalized)

        distances = np.sort(distances[:, 4], axis=0)
        plt.plot(distances)
        plt.title("Wykres k-distance")
        plt.ylabel("Odległość do 5-tego sąsiada (Eps)")
        plt.show()

class FaceClassifier:
    """Cluster unlabeled faces and run multi-class SVM classification."""

    def __init__(self):
        """Initialize classifier state."""
        self.svm_model = None
        self.is_trained = False

    def get_face_clusters(self, embeddings: np.ndarray, fids: list) -> dict:
        """Group unlabeled embeddings using DBSCAN with cosine distance."""

        dbscan = DBSCAN(eps=0.35, min_samples=2, metric="cosine")

        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for fid, label in zip(fids, labels):
            if label == -1:  # noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fid)
        return clusters

    def train_one_vs_rest_svm(self, x_train: list, y_train_labels: list) -> None:
        """Train an One-vs-Rest SVM pipeline with grid search."""
        if len(set(y_train_labels)) < 2:
            return

        # pipe = Pipeline(
        #     [
        #         ("pca", PCA(n_components=150)),
        #         ("clf", OneVsRestClassifier(SVC(kernel="rbf", probability=True, class_weight="balanced"), n_jobs = -1)),
        #     ]
        # )

        # param_grid = {
        #     "clf__estimator__C": [0.1, 1, 10, 100],
        #     "clf__estimator__gamma": [0.001, 0.01, 0.1, "scale"],
        # }

        # search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)

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

        self.svm_model = search.best_estimator_
        self.is_trained = True

        print(f"Best parameters (OvR): {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.2%}")

    def predict_unlabeled(self, x_test: list) -> tuple:
        """Predict labels and confidence scores for unlabeled embeddings."""
        if not self.is_trained:
            return [], []

        predictions = self.svm_model.predict(x_test)
        probabilities = np.max(self.svm_model.predict_proba(x_test), axis=1)

        return predictions, probabilities
