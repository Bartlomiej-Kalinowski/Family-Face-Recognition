import os

class Config:
    """Project configuration constants."""
    YOLO_MODEL_PATH = './face_detection_model/yolov8n-face.pt'
    SOURCE_DIR = "./WIDER_train" # "./datasetFIW/train"
    OUTPUT_DIR = "./output_data"

    ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_originals")
    FACES_DIR = os.path.join(OUTPUT_DIR, "extracted_faces")
    DB_PATH = os.path.join(OUTPUT_DIR, "face_metadata.db")

    FACE_SIZE = (128, 128)
    CONFIDENCE_THRESHOLD = 0.5
    MATCH_PROBABILITY_THRESHOLD = 0.5

    @classmethod
    def update_source_dir(cls, new_path):
        """Dynamically updates the source directory."""
        if os.path.exists(new_path):
            cls.SOURCE_DIR = new_path