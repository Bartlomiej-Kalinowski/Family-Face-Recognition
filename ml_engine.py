"""Machine-learning components for face extraction, clustering, and classification."""
import math
import sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from torch import optim

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
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F



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
            print(f"Brak sciezki do pliku z twarza: {image_path}")
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

        # Nadpisanie oryginalnego pliku na dysku
        success = cv2.imwrite(image_path, aligned_img)
        if not success:
            print(f"[OSTRZEŻENIE] Nie udało się nadpisać pliku: {image_path}")
        # =========================================================

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


    def compute_embedding_from_crop(self, alignment = False):
        nb_features = None
        self.db.clear_embeddings(self.dataset)
        updated_correctly = 0
        if alignment == True:
            self.recognizer = cv2.FaceRecognizerSF.create(
                self.config.FACE_RECOGNIZER_CV_PATH,
                ""
            )
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

        #WERSJA 2: bez face alignment
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



class FaceClusterer:
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


class SVMClassifier:
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

    def predict_unlabeled(self, x_test: np.ndarray, threshold = 0.3) -> tuple:
        """Predict labels and confidence scores for unlabeled embeddings."""
        if not self.is_trained:
            return [], []

        probs_all = self.model.predict_proba(x_test)

        # Wyciągamy najwyższe prawdopodobieństwo i odpowiadający mu indeks klasy
        max_probs = np.max(probs_all, axis=1)
        max_indices = np.argmax(probs_all, axis=1)

        final_predictions = []
        class_names = self.model.classes_

        for i in range(len(max_probs)):
            prob = max_probs[i]
            if prob < threshold:
                final_predictions.append("Nieznana osoba")
            else:
                final_predictions.append(class_names[max_indices[i]])
            print(f"Oryginał: {class_names[max_indices[i]]}, Pewność: {prob:.2%}")

        return np.array(final_predictions), max_probs


class KNNclassifier:
    """KNN classifier for face classification."""

    def __init__(self, x_train: np.ndarray, y_train: list[str], distance_threshold: float = 100):
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
        self.kdt = BallTree(x_train, leaf_size=30)

    def predict_unlabeled(self, x_test: np.ndarray) -> tuple:
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

        print("Typ predykcji zwracanej przez KNN: ", type(predictions))
        return predictions, None


class VGGClassifier(nn.Module):
    """VGG classifier for face classification."""
    def __init__(self, num_classes, idx_to_class ,num_epochs_ = 10):
        super(VGGClassifier, self).__init__()
        self.vgg16_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT) # loads pretrained weights
        for param in self.vgg16_model.parameters(): # freezes all convolutional layers
            param.requires_grad = False
        in_features = self.vgg16_model.classifier[6].in_features # classifier[6] is the last fully connected layer
        self.vgg16_model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.6),  # Silny dropout dla małego zbioru
            nn.Linear(in_features, num_classes)  # one linear layer - small train set
        )
        self.num_classes = num_classes
        self.num_epochs = num_epochs_
        self.idx_to_class = idx_to_class

    def prepare_data(self, x_train, y_train, batch_size=16):
        # 1. Konwersja na Tensory
        # x_train: [liczba_zdjec, kanały, wysokość, szerokość]
        # y_train: [liczba_zdjec] (liczby całkowite)
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # 2. Stworzenie obiektu Dataset (paczka cechy + etykiety)
        dataset = TensorDataset(x_tensor, y_tensor)

        # 3. Stworzenie DataLoadera
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader

    def forward(self, x):
        # Definiujemy jak dane płyną przez sieć
        return self.vgg16_model(x)

    def fit(self, x_train, y_train):
        criterion = nn.CrossEntropyLoss()
        # Optymalizator widzi tylko parametry z requires_grad=True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.vgg16_model.parameters()), lr=0.0001)

        # 2. Pętla treningowa
        train_loader = self.prepare_data(x_train, y_train)
        self.vgg16_model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.vgg16_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(train_loader)}")

    def predict_unlabeled(self, x_test, threshold = 0.25):
        self.vgg16_model.eval() # test mode, without dropout
        predicted_names = []

        x_tensor = torch.tensor(x_test, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(x_tensor), batch_size=32, shuffle=False)


        with torch.no_grad():  # nie liczymy gradientow podczas testu
            for inputs in tqdm(test_loader, "Ocena skutecznosci modelu"):
                outputs = self.vgg16_model(inputs[0])
                probabilities = F.softmax(outputs, dim=1)

                # 2. Pobranie najwyższego prawdopodobieństwa i jego indeksu
                max_probs, predicted_indices = torch.max(probabilities, 1)

                # 3. Sprawdzanie progu (Threshold) dla każdego zdjęcia
                for prob, idx in zip(max_probs, predicted_indices):
                    # Sprawdzamy, czy model jest pewny swego
                    if prob.item() >= threshold:
                        name = self.idx_to_class[idx.item()]
                    else:
                        name = "Nieznany"  # Model jest zbyt niepewny

                    predicted_names.append((name, prob))

            return predicted_names






