from ultralytics import YOLO
import numpy as np
import cv2
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN


class FaceEngine:
    """Manages YOLO detection and identity matching logic without console interference."""

    def __init__(self, config):
        """Loads YOLOv8-face model and configuration."""
        # Automatyczny wybór urządzenia

        self.config = config
        self.detector = YOLO(config.YOLO_MODEL_PATH)

    def extract_face_data(self, image_path):
        """Detects faces and extracts their feature embeddings with padding and high-quality crops."""
        img = cv2.imread(image_path)
        if img is None:
            return []

        results = self.detector(image_path, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        faces_data = []
        img_h, img_w, _ = img.shape

        # Inicjalizacja CLAHE (lepsze niż equalizeHist, bo działa lokalnie)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Inicjalizacja HOG (wyciąga krawędzie, ignoruje oświetlenie globalne)
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

        for i, box in enumerate(results.boxes):
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords

            # 1. DODAWANIE MARGINESU (Padding 20%)
            # Zapobiega "ciasnym" i niewyraźnym cropom
            w_box = x2 - x1
            h_box = y2 - y1
            padding = 0.2

            x1 = max(0, int(x1 - w_box * padding))
            y1 = max(0, int(y1 - h_box * padding))
            x2 = min(img_w, int(x2 + w_box * padding))
            y2 = min(img_h, int(y2 + h_box * padding))

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0: continue

            # 2. WYSOKA JAKOŚĆ WYCINKA DLA GUI
            # Resizing z interpolacją Lanczos dla lepszej ostrości
            # Możesz zwiększyć do 256, jeśli chcesz mieć bardzo wyraźne miniatury
            display_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)

            # 3. PREPROCESSING DLA EMBEDDINGU (ZMIANA NA HOG)
            resized_gray = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized_gray, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)

            # Obliczanie wektora cech HOG
            embedding = hog.compute(gray).flatten().astype(np.float32)

            # Normalizacja wektora (kluczowe dla metryki cosine w DBSCAN)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            faces_data.append({
                "crop": display_crop,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "embedding": embedding.tolist()
            })

        return faces_data


    def get_face_clusters(self, embeddings, fids):
        # eps: odległość (kluczowy parametr, zacznij od 0.5)
        # min_samples: ile zdjęć musi być blisko siebie, by uznać to za grupę
        dbscan = DBSCAN(eps=0.40, min_samples=3, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for fid, label in zip(fids, labels):
            if label == -1: continue  # -1 to szum (zdjęcia niepasujące do nikogo)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fid)
        return clusters

    def find_similar_unlabeled(self, anchor_fid, db, top_k=20):
        """Wyszukiwanie punktowe po kliknięciu na jedną twarz."""
        anchor_meta = db.get_metadata_for_gui(anchor_fid)
        if not anchor_meta or 'embedding' not in anchor_meta:
            return []

        anchor_emb = np.array(anchor_meta['embedding']).reshape(1, -1)
        unlabeled_data = db.get_all_unlabeled_embeddings()

        if not unlabeled_data: return []

        fids = [item[0] for item in unlabeled_data]
        embeddings = np.array([item[1] for item in unlabeled_data])

        # Obliczamy dystans
        distances = np.linalg.norm(embeddings - anchor_emb, axis=1)

        # Argsort zwraca indeksy od najmniejszego dystansu
        sorted_indices = np.argsort(distances)

        # Zwracamy FIDy (pomijamy pierwszy, bo to prawdopodobnie ta sama fotka co anchor)
        return [fids[i] for i in sorted_indices[:top_k]]

    def train_and_match(self, anchor_embeddings, candidate_embeddings):
        """Trenuje OCSVM na zestawie wzorców."""
        # nu=0.1 pozwala na 10% 'odrzutów' w grupie treningowej (odporność na błędy)
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        clf.fit(anchor_embeddings)

        scores = clf.decision_function(candidate_embeddings)
        # Sigmoidalna normalizacja wyników do zakresu 0-1
        probs = 1 / (1 + np.exp(-scores))
        return probs