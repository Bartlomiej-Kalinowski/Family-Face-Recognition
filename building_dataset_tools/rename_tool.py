"""Interactive helper for assigning labels and renaming dataset images."""

import os
from collections import defaultdict

import cv2


def label_and_rename_images(folder_path: str):
    """Preview images, collect labels, and rename files to `<label>_<index>.ext`."""
    valid_extensions = (".png", ".jpg", ".jpeg")

    if not os.path.exists(folder_path):
        print(f"Błąd: Folder '{folder_path}' nie istnieje.")
        return

    images = [f_name for f_name in os.listdir(folder_path) if f_name.lower().endswith(valid_extensions)]
    if not images:
        print("Brak zdjęć w folderze.")
        return

    label_counts = defaultdict(int)

    print(f"Znaleziono {len(images)} zdjęć.")
    print("Instrukcja: Wpisz imię/nazwisko. Wpisz 's' by pominąć, 'q' by wyjść.")

    for img_name in images:
        old_path = os.path.join(folder_path, img_name)
        img = cv2.imread(old_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        max_size = 800

        # Downscale large previews to keep images inside typical screen bounds.
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_view = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_view = img

        cv2.imshow("Oryginalne Zdjecie (Wpisz etykiete w konsoli)", img_view)
        cv2.waitKey(1)

        label = input(f"Kto jest na zdjęciu '{img_name}'? > ").strip()

        if label.lower() == "q":
            break
        if label.lower() == "s" or not label:
            print("  [-] Pominięto.")
            continue

        safe_label = label.replace(" ", "_")

        label_counts[safe_label] += 1
        number = label_counts[safe_label]

        ext = os.path.splitext(img_name)[1].lower()
        new_name = f"{safe_label}_{number:02d}{ext}"
        new_path = os.path.join(folder_path, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"  [+] Zmieniono na: {new_name}")
        except Exception as e:
            print(f"  [!] Błąd podczas zmiany nazwy: {e}")

    cv2.destroyAllWindows()
    print("Zakończono zmianę nazw plików.")


if __name__ == "__main__":
    folder = input("Podaj ścieżkę do folderu ze zdjęciami: ")
    label_and_rename_images(folder)
