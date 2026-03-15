"""Terminal fallback utility for manual labeling of extracted face crops."""

import os

import cv2

from config import Config
from database import FaceDatabase


def run_manual_labeler():
    """Iterate over unlabeled faces and store labels entered in the console."""
    config = Config()
    db = FaceDatabase(config)

    unlabeled_rows = db.get_all_unlabeled_embeddings()
    if not unlabeled_rows:
        print("Wszystkie twarze są podpisane.")
        return

    for face_id, _ in unlabeled_rows:
        face_path = os.path.join(config.FACES_DIR, f"{face_id}.jpg")
        if not os.path.exists(face_path):
            continue

        img = cv2.imread(face_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (256, 256))
        cv2.imshow("ID: " + face_id, img_resized)
        cv2.waitKey(1)

        label = input(f"Podpis dla {face_id} > ").strip()
        cv2.destroyAllWindows()

        if label.lower() == "q":
            break
        if label and label.lower() != "s":
            db.set_manual_label(face_id, label)
            print(f"Zapisano: {label}")


if __name__ == "__main__":
    run_manual_labeler()
