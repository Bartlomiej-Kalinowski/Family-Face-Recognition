"""Project-wide configuration values and filesystem paths."""

import os
from pathlib import Path
import shutil


class Config:
    """Store global configuration constants used by the application."""

    YOLO_MODEL_PATH = "./face_detection_model/yolov8n-face.pt"

    BASE_DIR = Path(__file__).resolve().parent
    SOURCE_DIR = str(BASE_DIR / "da_images")
    OUTPUT_DIR = str(BASE_DIR / "output_data")

    ANNOTATED_FACES_DIR = os.path.join(OUTPUT_DIR, "annotated_faces")
    FACES_DIR = os.path.join(OUTPUT_DIR, "extracted_faces")
    DB_PATH = os.path.join(OUTPUT_DIR, "face_metadata.db")

    FACE_SIZE = (128, 128)
    CONFIDENCE_THRESHOLD = 0.5
    MATCH_PROBABILITY_THRESHOLD = 0.5

    # Recreate the visualization output directory on each startup.
    if os.path.exists(ANNOTATED_FACES_DIR):
        shutil.rmtree(ANNOTATED_FACES_DIR)

    os.makedirs(ANNOTATED_FACES_DIR)

    @classmethod
    def update_source_dir(cls, new_path: str) -> None:
        """Update the source image directory when the provided path exists."""
        if os.path.exists(new_path):
            cls.SOURCE_DIR = new_path
