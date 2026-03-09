from config import Config
from database import FaceDatabase

def fix_label():
    """Pozwala na poprawę literówek w bazie na podstawie podanego ID pliku."""
    db = FaceDatabase(Config())

    while True:
        face_id = input("\nPodaj ID (lub 'exit'): ").strip().replace(".jpg", "")
        if face_id.lower() == 'exit': break

        current_label = db.get_label_by_id(face_id)
        if current_label:
            print(f"Aktualna etykieta: '{current_label}'")
            new_label = input("Nowa etykieta: ").strip()
            if new_label:
                db.set_label(face_id, new_label, validated=True)
                print("Zaktualizowano.")
        else:
            print("Nie znaleziono w bazie.")

if __name__ == "__main__":
    fix_label()