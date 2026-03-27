"""Machine-learning components for face extraction, clustering, and classification."""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC



class FaceExtractor:
    """Detect faces with YOLO and compute HOG embeddings for each crop."""

    def __init__(self, config):
        from ultralytics import YOLO
        """Load the YOLO detector using paths from the config object."""
        self.config = config
        self.detector = YOLO(config.YOLO_MODEL_PATH)

    @staticmethod
    def compute_embedding_from_crop( face_crop: np.ndarray) -> np.ndarray:
        """Oblicza NOWY embedding HOG dla wyciętej twarzy."""
        if face_crop.size == 0:
            return None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

        gray = cv2.cvtColor(
            cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        gray = clahe.apply(gray)

        embedding = np.array(hog.compute(gray)).flatten().astype(np.float32)
        # Normalizacja L2
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        return embedding

    def extract_face_data(self, image_path: str) -> list:
        """Return detected faces as dictionaries with crop, bbox, and embedding."""
        img = cv2.imread(image_path)
        if img is None:
            return []

        results = self.detector(image_path, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        faces_data = []
        img_h, img_w, _ = img.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Add context around the face to make embeddings less brittle.
            w_box, h_box = x2 - x1, y2 - y1
            pad = 0.0
            x1, y1 = max(0, int(x1 - w_box * pad)), max(0, int(y1 - h_box * pad))
            x2, y2 = min(img_w, int(x2 + w_box * pad)), min(img_h, int(y2 + h_box * pad))

            face_crop = img[y1:y2, x1:x2]

            # Zamiast pisać logikę HOG tutaj, wywołaj nową metodę:
            embedding = self.compute_embedding_from_crop(face_crop)

            if embedding is not None:
                faces_data.append({
                    "original_image": image_path,
                    "crop": face_crop,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "embedding": embedding.tolist(),
                })
        return faces_data

class FaceClassifier:
    """Cluster unlabeled faces and run multi-class SVM classification."""

    def __init__(self):
        """Initialize classifier state."""
        self.svm_model = None
        self.is_trained = False

    def get_face_clusters(self, embeddings: np.ndarray, fids: list) -> dict:
        """Group unlabeled embeddings using DBSCAN with cosine distance."""

        dbscan = DBSCAN(eps=0.31, min_samples=5, metric="cosine")
        pca = PCA(n_components=0.6)

        embeddings = pca.fit_transform(embeddings)

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

        pipe = Pipeline(
            [
                ("pca", PCA(n_components=0.7)),
                ("clf", OneVsRestClassifier(SVC(kernel="rbf", probability=True, class_weight="balanced"))),
            ]
        )

        param_grid = {
            "clf__estimator__C": [0.1, 1, 10, 100],
            "clf__estimator__gamma": [0.001, 0.01, 0.1, "scale"],
        }

        search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
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
