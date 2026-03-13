import os
from pathlib import Path
import shutil


class Config:
    """
    Przechowuje globalne ustawienia projektu i stałe konfiguracyjne.
    Zasada SRP: Ta klasa odpowiada TYLKO za trzymanie wartości konfiguracyjnych.
    """
    YOLO_MODEL_PATH = './face_detection_model/yolov8n-face.pt'

    # Używamy ścieżek względnych lub domyślnego folderu, aby uniknąć błędów na innych komputerach
    BASE_DIR = Path(__file__).resolve().parent
    SOURCE_DIR = str(BASE_DIR / "da_images")  # Domyślny folder ze zdjęciami
    OUTPUT_DIR = str(BASE_DIR / "output_data")

    ANNOTATED_FACES_DIR = os.path.join(OUTPUT_DIR, "annotated_faces")
    FACES_DIR = os.path.join(OUTPUT_DIR, "extracted_faces")
    DB_PATH = os.path.join(OUTPUT_DIR, "face_metadata.db")

    FACE_SIZE = (128, 128)
    CONFIDENCE_THRESHOLD = 0.5
    MATCH_PROBABILITY_THRESHOLD = 0.5

    if os.path.exists(ANNOTATED_FACES_DIR):
        shutil.rmtree(ANNOTATED_FACES_DIR)  # Usuwa folder i wszystko w środku

    os.makedirs(ANNOTATED_FACES_DIR)

    @classmethod
    def update_source_dir(cls, new_path: str) -> None:
        """Dynamicznie aktualizuje ścieżkę do folderu źródłowego."""
        if os.path.exists(new_path):
            cls.SOURCE_DIR = new_path