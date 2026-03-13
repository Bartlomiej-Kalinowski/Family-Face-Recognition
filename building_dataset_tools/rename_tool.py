import os
import cv2
from collections import defaultdict


def label_and_rename_images(folder_path: str):
    valid_extensions = ('.png', '.jpg', '.jpeg')

    if not os.path.exists(folder_path):
        print(f"Błąd: Folder '{folder_path}' nie istnieje.")
        return

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    if not images:
        print("Brak zdjęć w folderze.")
        return

    # Słownik do śledzenia numeracji (np. {'Jan_Kowalski': 2})
    label_counts = defaultdict(int)

    print(f"Znaleziono {len(images)} zdjęć.")
    print("Instrukcja: Wpisz imię/nazwisko. Wpisz 's' by pominąć, 'q' by wyjść.")

    for img_name in images:
        # Jeśli zdjęcie już wygląda na podpisane (np. Jan_Kowalski_01.jpg), możemy je pominąć
        # Zakładamy, że niepodpisane zdjęcia to np. DSC001.jpg lub IMG_123.jpg

        old_path = os.path.join(folder_path, img_name)
        img = cv2.imread(old_path)
        if img is None:
            continue

        # Skalowanie podglądu, by zdjęcie nie wychodziło poza ekran
        h, w = img.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_view = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_view = img

        cv2.imshow("Oryginalne Zdjecie (Wpisz etykiete w konsoli)", img_view)
        cv2.waitKey(1)

        # Pobieranie etykiety
        label = input(f"Kto jest na zdjęciu '{img_name}'? > ").strip()

        if label.lower() == 'q':
            break
        if label.lower() == 's' or not label:
            print("  [-] Pominięto.")
            continue

        # Zamiana spacji na podkreślniki
        safe_label = label.replace(" ", "_")

        # Inkrementacja licznika dla danej osoby
        label_counts[safe_label] += 1
        number = label_counts[safe_label]

        # Tworzenie nowej nazwy
        ext = os.path.splitext(img_name)[1].lower()
        # np. Jan_Kowalski + _ + 01 + .jpg
        new_name = f"{safe_label}_{number:02d}{ext}"
        new_path = os.path.join(folder_path, new_name)

        # Zmiana nazwy pliku na dysku
        try:
            os.rename(old_path, new_path)
            print(f"  [+] Zmieniono na: {new_name}")
        except Exception as e:
            print(f"  [!] Błąd podczas zmiany nazwy: {e}")

    cv2.destroyAllWindows()
    print("Zakończono zmianę nazw plików.")


if __name__ == "__main__":
    # Możesz tu wpisać ścieżkę na sztywno, np. "C:/MojeZdjecia/"
    folder = input("Podaj ścieżkę do folderu ze zdjęciami: ")
    label_and_rename_images(folder)