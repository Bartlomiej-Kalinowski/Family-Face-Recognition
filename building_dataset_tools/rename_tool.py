"""Interactive helper for labeling/renaming cropped faces with DB sync."""

import os
import re
from collections import defaultdict

import cv2

from config import Config
from database import FaceDatabase


def _sanitize_label(label: str) -> str:
    """Keep only filename-safe chars for generated names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")


def _preview_image(path: str) -> bool:
    """Show image preview. Returns False if file cannot be read."""
    img = cv2.imread(path)
    if img is None:
        return False

    h, w = img.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow("Cropped face (label in console)", img)
    cv2.waitKey(1)
    return True


def _rename_file_and_sync_db(
    db: FaceDatabase,
    old_path: str,
    new_path: str,
    old_face_id: str,
    new_face_id: str,
) -> None:
    """Rename file and try to sync `face_id` + `image_path` in DB."""
    os.rename(old_path, new_path)
    status = db.rename_face_record(old_face_id, new_face_id, os.path.abspath(new_path))

    if status == "updated":
        print(f"  [+] Renamed: {os.path.basename(new_path)} (DB updated)")
        return

    if status == "missing_old":
        print(f"  [i] Renamed: {os.path.basename(new_path)} (no DB row for '{old_face_id}')")
        return

    os.rename(new_path, old_path)
    print(f"  [!] DB collision for '{new_face_id}', restored original file name.")


def label_and_rename_images(folder_path: str, db: FaceDatabase) -> None:
    """Bulk mode: generate names as `<label>_<index>.ext`."""
    valid_extensions = (".png", ".jpg", ".jpeg")
    if not os.path.exists(folder_path):
        print(f"Error: folder '{folder_path}' does not exist.")
        return

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    if not images:
        print("No images found in folder.")
        return

    label_counts = defaultdict(int)
    print(f"Found {len(images)} images.")
    print("Type label, 's' to skip, 'q' to quit.")

    for img_name in images:
        old_path = os.path.join(folder_path, img_name)
        if not _preview_image(old_path):
            continue

        label = input(f"Who is in '{img_name}'? > ").strip()
        if label.lower() == "q":
            break
        if label.lower() == "s" or not label:
            print("  [-] Skipped.")
            continue

        safe_label = _sanitize_label(label)
        if not safe_label:
            print("  [!] Invalid label after sanitization.")
            continue

        label_counts[safe_label] += 1
        index = label_counts[safe_label]
        ext = os.path.splitext(img_name)[1].lower()

        new_name = f"{safe_label}_{index:02d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        if os.path.exists(new_path):
            print(f"  [!] Target exists: {new_name}")
            continue

        old_face_id = os.path.splitext(img_name)[0]
        new_face_id = os.path.splitext(new_name)[0]

        try:
            _rename_file_and_sync_db(db, old_path, new_path, old_face_id, new_face_id)
        except Exception as exc:
            print(f"  [!] Rename failed: {exc}")

    cv2.destroyAllWindows()
    print("Done.")


def label_one_image(db: FaceDatabase) -> None:
    """Manual mode: rename one selected file to `<label>.ext`."""
    img_path = input("Path to image file: ").strip()
    if not os.path.exists(img_path):
        print(f"Error: '{img_path}' does not exist.")
        return
    if not _preview_image(img_path):
        print("Error: cannot read image.")
        return

    label = input("New label (filename stem) > ").strip()
    safe_label = _sanitize_label(label)
    if not safe_label:
        print("Invalid label.")
        return

    folder = os.path.dirname(img_path)
    ext = os.path.splitext(img_path)[1].lower() or ".jpg"
    new_name = f"{safe_label}{ext}"
    new_path = os.path.join(folder, new_name)
    if os.path.exists(new_path):
        print(f"Target exists: {new_path}")
        return

    old_face_id = os.path.splitext(os.path.basename(img_path))[0]
    new_face_id = os.path.splitext(new_name)[0]

    try:
        _rename_file_and_sync_db(db, img_path, new_path, old_face_id, new_face_id)
    except Exception as exc:
        print(f"Rename failed: {exc}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = Config()
    db = FaceDatabase(config)

    folder = input(f"Cropped-faces folder (Enter = {config.FACES_DIR}): ").strip()
    target_folder = folder or config.FACES_DIR
    mode = input("Mode: 'f' (bulk) or 'm' (single file) > ").strip().lower()

    try:
        if mode == "f":
            label_and_rename_images(target_folder, db)
        elif mode == "m":
            label_one_image(db)
        else:
            print("Unknown mode.")
    finally:
        db.close()
