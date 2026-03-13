from ultralytics import YOLO
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


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
            embedding = np.array(hog.compute(gray)).flatten().astype(np.float32)
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
        self.is_trained = False

    def get_face_clusters(self, embeddings: np.ndarray, fids: list) -> dict:
        """Grupuje niepodpisane twarze w klastry używając algorytmu DBSCAN."""
        dbscan = DBSCAN(eps=0.1, min_samples=3, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for fid, label in zip(fids, labels):
            # if label == -1: continue  # Odrzucamy szum
            if label not in clusters: clusters[label] = []
            clusters[label].append(fid)
        return clusters

    def train_multiclass_svm(self, X_train: list, y_train_labels: list) -> None:
        if len(set(y_train_labels)) < 2:
            return

        # 1. Tworzymy Pipeline: Skalowanie -> PCA -> OneVsRest(SVC)
        # Zauważ, że SVC jest teraz wewnątrz OneVsRestClassifier
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('clf', OneVsRestClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced')))
        ])

        # 2. Definiujemy siatkę parametrów
        # UWAGA: Ponieważ SVC jest teraz "wnukiem" Pipeline'u (przez OneVsRest),
        # dostęp do parametrów mamy przez: nazwaStepu__estimator__parametr
        param_grid = {
            'clf__estimator__C': [0.1, 1, 10, 100],
            'clf__estimator__gamma': [0.001, 0.01, 0.1, 'scale']
        }

        # 3. Szukanie najlepszych ustawień
        search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
        search.fit(np.array(X_train), y_train_labels)

        # Zapisujemy cały Pipeline jako nasz model
        self.svm_model = search.best_estimator_
        self.is_trained = True

        print(f"Najlepsze parametry (OvR): {search.best_params_}")
        print(f"Najlepszy wynik (cross-val): {search.best_score_:.2%}")


    def predict_unlabeled(self, X_test: list) -> tuple:
        """
        Ocenia niepodpisane twarze. Zwraca przewidywane etykiety i pewność predykcji.
        """
        if not self.is_trained:
            return [], []

        predictions = self.svm_model.predict(X_test)
        probabilities = np.max(self.svm_model.predict_proba(X_test), axis=1)

        return predictions, probabilities