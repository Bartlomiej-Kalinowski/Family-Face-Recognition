"""Project-wide configuration values and filesystem paths."""

import os
import shutil
from pathlib import Path


class Config:
    """Store global configuration constants used by the application."""

    YOLO_MODEL_PATH = "./face_detection_model/yolov8n-face.pt"

    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = str(BASE_DIR / "output_data")
    SOURCE_DIR = str(BASE_DIR / "INPUT")

    ANNOTATED_FACES_DIR = os.path.join(OUTPUT_DIR, "annotated_faces")
    FACES_DIR = os.path.join(OUTPUT_DIR, "extracted_faces")
    DB_PATH = os.path.join(OUTPUT_DIR, "face_metadata.db")

    FACE_SIZE = (128, 128)
    CONFIDENCE_THRESHOLD = 0.5
    MATCH_PROBABILITY_THRESHOLD = 0.5
    _annotated_reset_done = False

    def __init__(self):
        """Ensure all runtime directories exist before the app starts."""
        self.ensure_required_dirs()
        self.reset_annotated_faces_dir()

    @classmethod
    def ensure_required_dirs(cls) -> None:
        """Create all folders needed by the application if they are missing."""
        required_dirs = [
            cls.SOURCE_DIR,
            cls.OUTPUT_DIR,
            cls.ANNOTATED_FACES_DIR,
            cls.FACES_DIR,
            os.path.dirname(cls.DB_PATH),
        ]
        for folder in required_dirs:
            if folder:
                os.makedirs(folder, exist_ok=True)

    @classmethod
    def reset_annotated_faces_dir(cls) -> None:
        """Recreate the visualization output directory on each startup."""
        if cls._annotated_reset_done:
            return

        if os.path.exists(cls.ANNOTATED_FACES_DIR):
            shutil.rmtree(cls.ANNOTATED_FACES_DIR)
        os.makedirs(cls.ANNOTATED_FACES_DIR, exist_ok=True)
        cls._annotated_reset_done = True

    @classmethod
    def update_source_dir(cls, new_path: str) -> None:
        """Update the source image directory when the provided path exists."""
        if os.path.isdir(new_path):
            cls.SOURCE_DIR = new_path
