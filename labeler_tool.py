import os
import cv2
from config import Config
from database import FaceDatabase

def run_manual_labeler():
    """
    Narzędzie terminalowe Fallback.
    Odtąd poprawnie komunikuje się z bazą poprzez metody dostępowe (DAO), a nie surowy SQL.
    """
    config = Config()
    db = FaceDatabase(config)

    unlabeled_ids = db.get_unlabeled_face_ids()
    if not unlabeled_ids: return print("Wszystkie twarze są podpisane.")

    for face_id in unlabeled_ids:
        face_path = os.path.join(config.FACES_DIR, f"{face_id}.jpg")
        if not os.path.exists(face_path): continue

        img_resized = cv2.resize(cv2.imread(face_path), (256, 256))
        cv2.imshow("ID: " + face_id, img_resized)
        cv2.waitKey(1)

        label = input(f"Podpis dla {face_id} > ").strip()
        cv2.destroyAllWindows()

        if label.lower() == 'q': break
        if label and label.lower() != 's':
            db.set_label(face_id, label, validated=True) # Zamiast db.cursor.execute
            print(f"Zapisano: {label}")

if __name__ == "__main__":
    run_manual_labeler()