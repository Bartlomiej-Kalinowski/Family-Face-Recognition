"""Interactive helper for labeling/renaming cropped faces with DB sync."""

import json
import os
import re
from collections import defaultdict

import cv2

from config import Config
from database import FaceDatabase

PROGRESS_FILE_NAME = ".label_progress.json"



def _sanitize_label(label: str) -> str:
    """Keep filename-safe characters for generated face IDs."""
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
    db1: FaceDatabase,
    old_path: str,
    new_path: str,
    old_face_id: str,
    new_face_id: str,
) -> None:
    """Rename file and try to sync `face_id` + `image_path` in DB."""
    status = db1.rename_face_record(old_face_id, new_face_id, os.path.abspath(new_path))

    if status == "updated":
        print(f"  [+] Renamed: {os.path.basename(new_path)} (DB updated)")
        os.rename(old_path, new_path)
        return

    if status == "missing_old":
        print(f"  [i] Renamed: {os.path.basename(new_path)} (no DB row for '{old_face_id}')")
        return

    os.rename(new_path, old_path)
    print(f"  [!] DB collision for '{new_face_id}', restored original file name.")


def _progress_path(folder_path: str) -> str:
    """Return the path to the per-folder labeling progress file."""
    return os.path.join(folder_path, PROGRESS_FILE_NAME)


def _load_progress(folder_path: str) -> dict | None:
    """Load progress from JSON if it exists and is valid."""
    path = _progress_path(folder_path)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(state, dict):
        return None
    if not isinstance(state.get("remaining_files"), list):
        return None
    if not isinstance(state.get("label_counts"), dict):
        return None

    return state


def _save_progress(folder_path: str, remaining_files: list[str], label_counts: dict[str, int]) -> None:
    """Persist current progress so labeling can continue later."""
    state = {
        "folder_path": os.path.abspath(folder_path),
        "remaining_files": remaining_files,
        "label_counts": label_counts,
    }
    with open(_progress_path(folder_path), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _clear_progress(folder_path: str) -> None:
    """Remove saved progress after completion or reset."""
    path = _progress_path(folder_path)
    if os.path.exists(path):
        os.remove(path)


def label_and_rename_images(folder_path: str, db: FaceDatabase) -> None:
    """Bulk mode: generate names as `<label>_<index>.ext`."""
    valid_extensions = (".png", ".jpg", ".jpeg")
    if not os.path.exists(folder_path):
        print(f"Error: folder '{folder_path}' does not exist.")
        return

    images = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions))
    saved_state = _load_progress(folder_path)

    label_counts = defaultdict(int)
    if saved_state:
        decision = input("Found saved progress. Type 'c' to continue or 'r' to reset > ").strip().lower()
        if decision == "c":
            remaining = [f for f in saved_state["remaining_files"] if os.path.exists(os.path.join(folder_path, f))]
            for key, value in saved_state["label_counts"].items():
                try:
                    label_counts[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            images = remaining
            print(f"Resumed session. Remaining images: {len(images)}")
        elif decision == "r":
            _clear_progress(folder_path)
            print("Progress reset. Starting from scratch.")
        else:
            print("Unknown option, starting from scratch.")
            _clear_progress(folder_path)

    if not images:
        print("No images found in folder.")
        _clear_progress(folder_path)
        return

    print(f"Images in current session: {len(images)}")
    print("Type label, 's' to skip, 'q' to save and quit.")

    for idx, img_name in enumerate(images):
        old_path = os.path.join(folder_path, img_name)
        remaining_after_current = images[idx + 1 :]

        if not _preview_image(old_path):
            _save_progress(folder_path, remaining_after_current, dict(label_counts))
            continue

        label = input(f"Who is in '{img_name}'? > ").strip()
        if label.lower() == "q":
            _save_progress(folder_path, images[idx:], dict(label_counts))
            print("Progress saved. You can resume later.")
            break
        if label.lower() == "s" or not label:
            print("  [-] Skipped.")
            _save_progress(folder_path, remaining_after_current, dict(label_counts))
            continue

        safe_label = _sanitize_label(label)
        if not safe_label:
            print("  [!] Invalid label after sanitization.")
            _save_progress(folder_path, remaining_after_current, dict(label_counts))
            continue

        label_counts[safe_label] += 1
        index = label_counts[safe_label]
        ext = os.path.splitext(img_name)[1].lower()

        new_name = f"{safe_label}_{index:02d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        if os.path.exists(new_path):
            print(f"  [!] Target exists: {new_name}")
            _save_progress(folder_path, remaining_after_current, dict(label_counts))
            continue

        old_face_id = os.path.splitext(img_name)[0]
        new_face_id = os.path.splitext(new_name)[0]

        try:
            _rename_file_and_sync_db(db, old_path, new_path, old_face_id, new_face_id)
        except Exception as exc:
            print(f"  [!] Rename failed: {exc}")
        finally:
            _save_progress(folder_path, remaining_after_current, dict(label_counts))
    else:
        _clear_progress(folder_path)
        print("Session complete. Progress file removed.")

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
