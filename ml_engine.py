"""Machine-learning components for face extraction, clustering, and classification."""
import copy
import math
import sys
from collections import Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from torch import optim

from database import FaceDatabase

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    from ultralytics import YOLO

    _ = torch.empty(1)
except Exception:
    print("Blad importu biblioteki ultralytics lub pytorch")
    exit(0)

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import BallTree
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader, Subset
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

        # # Pobieramy wymiary oryginalnego obrazu
        # h, w = img.shape[:2]
        #
        # # Obliczamy margines (5% z każdej strony)
        # margin_h = int(h * 0.025)
        # margin_w = int(w * 0.025)
        #
        # # Przycinamy obraz (Crop)
        # # img[y_start : y_end, x_start : x_end]
        # img = img[margin_h: h - margin_h, margin_w: w - margin_w]

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
    def __init__(self):
        self.model = True
        self.is_trained = False


    def train_one_vs_rest_svm(self, x_train: list, y_train_labels: list) -> None:
        """Train an One-vs-Rest SVM pipeline with grid search."""
        print("X_train shape: ", len(x_train), "Y_train shape: ", len(y_train_labels))

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

        #-------do eksperymentow-----------------------------------------------------
        # pipe = Pipeline([
        #     ("clf", OneVsRestClassifier(SVC(probability=True, class_weight="balanced"), n_jobs=-1))
        # ])
        #
        # # 2. Siatka parametrów do sprawdzenia
        # param_grid = [
        #     {
        #         'clf__estimator__kernel': ['linear'],
        #         'clf__estimator__C': [0.1, 1, 10]
        #     },
        #     {
        #         'clf__estimator__kernel': ['rbf'],
        #         'clf__estimator__C': loguniform(1e3, 1e5),
        #         'clf__estimator__gamma': loguniform(1e-4, 1e-1)
        #     },
        #     {
        #         'clf__estimator__kernel': ['poly'],
        #         'clf__estimator__degree': [2, 3],
        #         'clf__estimator__C': [1, 10]
        #     }
        # ]
        # search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', verbose=2)
        #----------------------------------------------------------------------

        search.fit(x_train, y_train_labels)

        search.fit(np.array(x_train), y_train_labels)

        self.model = search.best_estimator_
        self.is_trained = True

        print(f"Best parameters (OvR): {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.2%}")

    def predict_unlabeled(self, x_test: np.ndarray, threshold = 0.3) -> tuple:
        """Predict labels and confidence scores for unlabeled embeddings."""

        print("X_test shape: ", x_test.shape, "Threshold: ", threshold)
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

    def __init__(self, x_train: tuple, y_train: list[str], distance_threshold: float = 1.0):
        """
        x_train: Macierz embeddingów wygenerowana i zatwierdzona z DBSCAN.
        y_train: Lista etykiet odpowiadająca wierszom w x_train (np. ["Jan", "Jan", "Anna", ...]).
        distance_threshold: Maksymalna odległość euklidesowa. Powyżej tego progu twarz to "Nieznany".
        """
        print("X_train shape: ", len(x_train), "Y_train shape: ", distance_threshold)

        self.y_train = np.array(y_train)
        self.threshold = distance_threshold
        self.k = 1


        # Budujemy drzewo tylko raz na zatwierdzonych danych
        print("[INFO] Budowanie BallTree...")
        self.bt = BallTree(x_train, leaf_size=30)

    def predict_unlabeled(self, x_test: np.ndarray) -> tuple:
        """
        Klasyfikuje nowe wektory twarzy na podstawie k najbliższych sąsiadów.
        """

        print("X_test shape: ", x_test.shape)
        from collections import Counter
        # Szukamy k-sąsiadów.
        distances, indices = self.bt.query(x_test, k=self.k, return_distance=True)
        predictions = []

        for i in range(len(x_test)):
            dist_i = distances[i]
            ind_i = indices[i]
            valid_neighbors_labels = []

            for j in range(self.k):
                current_dist = dist_i[j]
                print(current_dist)
                current_idx = ind_i[j]
                if current_dist <= self.threshold:
                    print("Current dist: ", current_dist)
                    valid_neighbors_labels.append(self.y_train[current_idx])

            if len(valid_neighbors_labels) == 0:
                print("Znalezino nieznana osobe")
                predictions.append("Nieznana osoba")
            else:
                # Głosowanie większościowe tylko wśród "bliskich" sąsiadów
                counts = Counter(valid_neighbors_labels)
                most_common_label = counts.most_common(1)[0][0]
                predictions.append(most_common_label)

        print("Typ predykcji zwracanej przez KNN: ", type(predictions), " ", predictions[0])
        return predictions, None


class VGGClassifier(nn.Module):

    def __init__(self, cf, num_classes, idx_to_class, num_epochs_=5):
        super().__init__()

        self.num_classes = num_classes

        self.model = InceptionResnetV1(pretrained='vggface2')

        # freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # classifier
        # self.classifier = nn.Linear(512, self.num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.idx_to_class = idx_to_class
        self.num_epochs = num_epochs_

    def forward(self, x):
        embeddings = self.model(x)      # [B, 512]
        logits = self.classifier(embeddings)  # [B, self.num_classes]
        return logits

    def prepare_data(self, x_train, y_train):

        print("X_train shape: ", len(x_train), "Y_train shape: ", len(y_train))
        # 1. Konwersja na Tensory
        # x_train: [liczba_zdjec, kanały, wysokość, szerokość]
        # y_train: [liczba_zdjec] (liczby całkowite)
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # 2. Stworzenie obiektu Dataset (paczka cechy + etykiety)
        dataset = TensorDataset(x_tensor, y_tensor)

        return dataset

    def fit(self, x_train, y_train, patience=3):

        # 1. Sztywny podział danych na Train (80%) i Val/Eval (20%) w pamięci RAM
        # Parametr stratify=should_stratify gwarantuje, że każda osoba będzie miała
        # reprezentację (zdjęcia) zarówno w zbiorze treningowym, jak i walidacyjnym.
        counter_dict = dict()
        unique_labels = set(y_train)
        for label in unique_labels:
            counter_dict[label] = 0
            for y in y_train:
                if y == label:
                    counter_dict[label] += 1
        min_samples = min(counter_dict.values())
        if min_samples < 2:
            should_stratify = None
        else:
            should_stratify = y_train

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.20, random_state=42, stratify=should_stratify
        )

        class_counts = Counter(y_tr)

        weights = torch.tensor(
            [1.0 / class_counts.get(i, 1) for i in range(self.num_classes)],
            dtype=torch.float32
        )

        criterion = nn.CrossEntropyLoss(weight=weights)

        # 2. Przygotowanie TensorDataset dla obu podzbiorów
        train_dataset = self.prepare_data(x_tr, y_tr)
        val_dataset = self.prepare_data(x_val, y_val)

        # 3. Stworzenie DataLoaderów
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 4. Optymalizator
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.005)

        # Zmienne pomocnicze do monitorowania Early Stopping i zapisu wag
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float('inf')
        patience_counter = 0  # Licznik epok bez poprawy błędu

        print(f"\nRozpoczynanie treningu (Maksymalna liczba epok: {self.num_epochs})")
        print(f"Rozmiar zbioru treningowego: {len(x_tr)} | Walidacyjnego: {len(x_val)}")
        print("-" * 60)

        for epoch in range(self.num_epochs):
            # --- faza treningu---
            self.model.train()  #  tryb treningowy
            running_train_loss = 0.0

            for inputs, labels in tqdm(train_loader, "Trenowanie modelu"):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)

            # --- faza walidacji ---
            self.model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, "Walidacja modelu"):
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            epoch_val_loss = running_val_loss / len(val_loader)

            # Wypisanie logów dla bieżącej epoki
            print(f"Epoch {epoch + 1:02d}/{self.num_epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}", end=" ")

            # early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0  # Reset licznika, bo znaleźliśmy lepszy punkt
                print(" -> [Zapisano najlepszy model]")
            else:
                patience_counter += 1  # Brak poprawy, zwiększamy licznik cierpliwości
                print(f" -> [Brak poprawy od {patience_counter} epok]")

            # Sprawdzenie warunku stopu
            if patience_counter >= patience:
                print(f"\n Early Stopping! Brak poprawy błędu walidacji przez {patience} kolejnych epok. Przerywam.")
                break

        # 5. Przywrócenie wag z momentu, w którym Val Loss był najniższy
        print(f"\nTrening zakończony. Przywracanie najlepszych wag (Najniższy Val Loss: {best_val_loss:.4f})")
        self.model.load_state_dict(best_model_wts)

    def predict_unlabeled(self, x_test, threshold=0.1):

        self.eval()
        predicted_names = []
        x_tensor = torch.tensor(x_test, dtype=torch.float32)
        test_loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=32,
            shuffle=False
        )

        with torch.no_grad():
            for inputs in test_loader:
                outputs = self.forward(inputs[0])
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predicted_indices = torch.max(probabilities, 1)
                for prob, idx in zip(max_probs, predicted_indices):
                    if prob.item() >= threshold:
                        name = self.idx_to_class[idx.item()]
                    else:
                        name = "Nieznana osoba"

                    predicted_names.append((name, prob.item()))

        return predicted_names








