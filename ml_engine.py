from ultralytics import YOLO
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class FaceExtractor:
    """
    Moduł Ekstrakcji Cech.
    Odpowiada za detekcję twarzy (YOLO) i zamianę obrazu na wektor liczbowy (HOG).
    Zasada SRP: Nie zajmuje się klasyfikacją ani bazą danych.
    """

    def __init__(self, config):
        """Inicjalizuje model detekcji YOLO na CPU/GPU."""
        self.config = config
        self.detector = YOLO(config.YOLO_MODEL_PATH)

    def extract_face_data(self, image_path: str) -> list:
        """Wykrywa twarze na obrazie, wycina je i generuje wektory HOG."""
        img = cv2.imread(image_path)
        if img is None: return []

        results = self.detector(image_path, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        faces_data = []
        img_h, img_w, _ = img.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Padding (margines)
            w_box, h_box = x2 - x1, y2 - y1
            pad = 0.2
            x1, y1 = max(0, int(x1 - w_box * pad)), max(0, int(y1 - h_box * pad))
            x2, y2 = min(img_w, int(x2 + w_box * pad)), min(img_h, int(y2 + h_box * pad))

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0: continue

            # Preprocessing GUI & HOG
            display_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)

            # Ekstrakcja cech (HOG)
            embedding = hog.compute(gray).flatten().astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            faces_data.append({
                "crop": display_crop,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "embedding": embedding.tolist()
            })

        return faces_data


class FaceClassifier:
    """
    Moduł Uczenia Maszynowego.
    Odpowiada za logikę grupowania (DBSCAN) i klasyfikacji nadzorowanej (Multi-class SVM).
    """

    def __init__(self):
        """Inicjalizuje globalny model klasyfikatora."""
        self.svm_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def get_face_clusters(self, embeddings: np.ndarray, fids: list) -> dict:
        """Grupuje niepodpisane twarze w klastry używając algorytmu DBSCAN."""
        dbscan = DBSCAN(eps=0.1, min_samples=3, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for fid, label in zip(fids, labels):
            if label == -1: continue  # Odrzucamy szum
            if label not in clusters: clusters[label] = []
            clusters[label].append(fid)
        return clusters

    def train_multiclass_svm(self, X_train: list, y_train_labels: list) -> None:
        """
        Trenuje globalny model One-vs-Rest SVM na wszystkich podpisanych twarzach.
        To całkowicie zastępuje poprzednie słowniki OneClassSVM!
        """
        if len(set(y_train_labels)) < 2:
            print("Zbyt mało unikalnych osób (wymagane min. 2), by wytrenować klasyfikator.")
            return

        # Standaryzacja wektorów
        X_scaled = self.scaler.fit_transform(np.array(X_train))

        # Wieloklasowy model SVM (C-Support Vector Classification) z jądrem RBF
        base_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.svm_model = OneVsRestClassifier(base_svm)

        self.svm_model.fit(X_scaled, y_train_labels)
        self.is_trained = True
        print(f"Pomyślnie wytrenowano Multi-class SVM na {len(X_train)} próbkach ({len(set(y_train_labels))} klas).")

    def predict_unlabeled(self, X_test: list) -> tuple:
        """
        Ocenia niepodpisane twarze. Zwraca przewidywane etykiety i pewność predykcji.
        """
        if not self.is_trained:
            return [], []

        X_test_scaled = self.scaler.transform(np.array(X_test))
        predictions = self.svm_model.predict(X_test_scaled)
        probabilities = np.max(self.svm_model.predict_proba(X_test_scaled), axis=1)

        return predictions, probabilities