import os
import cv2
from config import Config
from database import FaceDatabase


def run_manual_labeler():
    print("=== NARZĘDZIE DO RĘCZNEGO ETYKIETOWANIA ===")
    config = Config()
    db = FaceDatabase(config)

    # Pobieramy tylko te twarze, które nie mają jeszcze przypisanej etykiety
    db.cursor.execute("SELECT face_id FROM faces WHERE label IS NULL OR label = ''")
    unlabeled = db.cursor.fetchall()

    if not unlabeled:
        print("Świetnie! Wszystkie twarze w bazie są już podpisane.")
        return

    print(f"Pozostało twarzy do podpisania: {len(unlabeled)}\n")
    print("INSTRUKCJA:")
    print("1. Spójrz na otwarte okno ze zdjęciem.")
    print("2. Kliknij w okno KONSOLI (terminala) i wpisz imię.")
    print("3. Wciśnij ENTER, aby zapisać.")
    print(" - Wpisz 's', aby pominąć zdjęcie (np. rozmazane).")
    print(" - Wpisz 'q', aby zakończyć pracę i wyjść.\n")

    for row in unlabeled:
        face_id = row[0]
        # Zakładamy, że wycięte twarze zapisujesz z rozszerzeniem .jpg w folderze faces/
        face_path = os.path.join(config.FACES_DIR, f"{face_id}.jpg")

        if not os.path.exists(face_path):
            print(f"[UWAGA] Brak pliku na dysku: {face_path}")
            continue

        # Wczytujemy zdjęcie i powiększamy je, żeby było lepiej widać
        img = cv2.imread(face_path)
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Wyświetlamy okno z twarzą
        window_name = f"Kto to jest? (ID: {face_id})"
        cv2.imshow(window_name, img_resized)
        cv2.waitKey(1)  # Wymusza odświeżenie okna obrazu

        # Pobieramy imię z konsoli
        label = input(f"Podpis dla {face_id} > ").strip()

        cv2.destroyWindow(window_name)

        # Obsługa komend specjalnych
        if label.lower() == 'q':
            print("Przerywam pracę. Do zobaczenia!")
            break
        elif label.lower() == 's' or label == '':
            print(" -> Pominięto.")
            continue
        else:
            # Zapisujemy etykietę do bazy
            db.cursor.execute("UPDATE faces SET label = ? WHERE id = ?", (label, face_id))
            db.conn.commit()
            print(f" -> Zapisano: {label}")

    cv2.destroyAllWindows()
    print("\nEtykietowanie zakończone.")


if __name__ == "__main__":
    run_manual_labeler()