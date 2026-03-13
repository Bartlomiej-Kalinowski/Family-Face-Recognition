import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QScrollArea, QLineEdit, QFrame, QDialog, QGridLayout,
                             QProgressDialog, QCheckBox, QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QEventLoop, pyqtSignal
from config import Config


class FaceCard(QFrame):
    """
    Karta twarzy w głównym oknie - pokazuje wynik SVM i pozwala na poprawkę.
    """
    confirmed = pyqtSignal(str, str)

    def __init__(self, face_id, name, parent=None):
        super().__init__(parent)
        self.face_id = face_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(160)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; border: 1px solid #3e3e3e;")

        layout = QVBoxLayout()

        # 1. Obrazek twarzy
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        face_path = os.path.join(Config.FACES_DIR, f"{face_id}.jpg")
        pixmap = QPixmap(face_path)
        if not pixmap.isNull():
            self.lbl_img.setPixmap(pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(self.lbl_img)

        # 2. Pole imienia (wypełnione przez SVM)
        self.input_name = QLineEdit()
        self.input_name.setText(name)
        self.input_name.setStyleSheet(
            "padding: 5px; color: #ffffff; background-color: #1e1e1e; border: 1px solid #555; font-weight: bold;")
        self.input_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.input_name)

        # 3. Przycisk poprawy / potwierdzenia
        self.btn_confirm = QPushButton("Zmień / OK")
        self.btn_confirm.setStyleSheet("background-color: #444; color: white; padding: 4px;")
        self.btn_confirm.clicked.connect(self._on_confirm)
        layout.addWidget(self.btn_confirm)

        self.setLayout(layout)

    def _on_confirm(self):
        new_name = self.input_name.text().strip()
        self.confirmed.emit(self.face_id, new_name)
        self.setStyleSheet(
            "background-color: #1a3320; border: 1px solid #28a745;")  # Zmiana koloru na zielony po potwierdzeniu


class FaceInterface(QMainWindow):
    def __init__(self):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
            self._setup_dark_theme()

        super().__init__()
        self.setWindowTitle("Face Recognition - Panel Weryfikacji SVM")
        self.resize(1300, 900)
        self._init_ui()
        self.show()

    def _setup_dark_theme(self):
        self.app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.app.setPalette(palette)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Nagłówek
        header = QLabel("WYNIKI AUTOMATYCZNEJ KLASYFIKACJI (SVM)")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #007acc; margin: 10px;")
        layout.addWidget(header)

        # Obszar przewijania dla siatki twarzy
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #1e1e1e;")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll)

        # --- SEKCJA: PASEK POSTĘPU ---
        self.progress_container = QFrame()
        self.progress_container.setStyleSheet("background-color: #2b2b2b; border-radius: 5px;")
        progress_layout = QVBoxLayout(self.progress_container)

        self.statusLabel = QLabel("Gotowy")
        self.statusLabel.setStyleSheet("color: #aaa; font-size: 11px;")
        progress_layout.addWidget(self.statusLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setFixedHeight(15)
        self.progressBar.setTextVisible(True)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #444;
                        border-radius: 7px;
                        background-color: #1e1e1e;
                        text-align: center;
                        color: white;
                    }
                    QProgressBar::chunk {
                        background-color: #007acc;
                        border-radius: 6px;
                    }
                """)
        progress_layout.addWidget(self.progressBar)

        layout.addWidget(self.progress_container)  # Dodajemy kontener do głównego układu

        # Dolny pasek kontrolny
        controls = QHBoxLayout()
        btn_refresh = QPushButton("Odśwież widok")
        btn_refresh.clicked.connect(lambda: print("Refreshing..."))
        controls.addWidget(btn_refresh)

        self.lbl_stats = QLabel("Załadowane twarze: 0")
        controls.addStretch()
        controls.addWidget(self.lbl_stats)
        layout.addLayout(controls)

    # W interface.py
    def ask_for_scan_mode(self):
        msg = QMessageBox()
        msg.setWindowTitle("Tryb Skanowania")
        msg.setText("Wybierz sposób przygotowania bazy:")
        full_btn = msg.addButton("Pełny Skan (YOLO)", QMessageBox.ActionRole)
        incr_btn = msg.addButton("Przyrostowy", QMessageBox.ActionRole)
        exist_btn = msg.addButton("Mam już wycięte twarze", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Anuluj", QMessageBox.RejectRole)

        msg.exec_()

        if msg.clickedButton() == full_btn: return "full"
        if msg.clickedButton() == incr_btn: return "incremental"
        if msg.clickedButton() == exist_btn: return "use_existing"
        return "cancel"

    def refresh_classified_faces(self, face_data_list, callback):
        """
        Czyści siatkę i ładuje nowe twarze sklasyfikowane przez SVM.
        face_data_list: lista krotek (fid, name, is_manual)
        callback: funkcja w main.py obsługująca zmianę imienia
        """
        # Czyszczenie siatki
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        for i, (fid, name, *_) in enumerate(face_data_list):
            card = FaceCard(fid, name)
            card.confirmed.connect(callback)
            self.grid_layout.addWidget(card, i // 6, i % 6)

        self.lbl_stats.setText(f"Załadowane twarze: {len(face_data_list)}")

    def bulk_verify_faces(self, face_ids):
        """ Okno dialogowe dla DBSCAN z blokadą pustego imienia """
        self.final_selection = []
        self.bulk_name = ""
        dialog = QDialog(self)
        dialog.setWindowTitle("Wykryto nową grupę (DBSCAN / Mean Shift)")
        dialog.setMinimumSize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        # 1. Pole wprowadzania imienia
        self.bulk_input = QLineEdit()
        self.bulk_input.setPlaceholderText("Wpisz imię dla tej grupy (WYMAGANE)...")
        self.bulk_input.setStyleSheet("""
            padding: 10px; 
            font-size: 16px; 
            background-color: #333; 
            color: white; 
            border: 2px solid #555;
        """)
        dialog_layout.addWidget(self.bulk_input)

        # 2. Obszar z twarzami
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        inner_grid = QGridLayout(grid_widget)

        check_boxes = {}
        for i, fid in enumerate(face_ids):
            face_path = os.path.join(Config.FACES_DIR, f"{fid}.jpg")
            container = QVBoxLayout()
            img_label = QLabel()
            pixmap = QPixmap(face_path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            chk = QCheckBox("To ta osoba")
            chk.setChecked(True)
            container.addWidget(img_label, alignment=Qt.AlignCenter)
            container.addWidget(chk, alignment=Qt.AlignCenter)

            frame = QFrame()
            frame.setStyleSheet("background-color: #3d3d3d; border-radius: 5px;")
            frame.setLayout(container)
            inner_grid.addWidget(frame, i // 4, i % 4)
            check_boxes[fid] = chk

        scroll.setWidget(grid_widget)
        dialog_layout.addWidget(scroll)

        # 3. Przycisk zatwierdzenia z blokadą
        btn_confirm = QPushButton("Zatwierdź grupę")
        # Domyślnie wyłączamy przycisk i nadajemy mu styl "zablokowany"
        btn_confirm.setEnabled(False)
        btn_confirm.setStyleSheet("""
            QPushButton:disabled { background-color: #444; color: #888; }
            QPushButton:enabled { background-color: #28a745; color: white; font-weight: bold; }
            height: 40px;
        """)

        # Funkcja sprawdzająca czy pole nie jest puste
        def validate_bulk_input():
            is_valid = len(self.bulk_input.text().strip()) > 0
            btn_confirm.setEnabled(is_valid)
            if is_valid:
                self.bulk_input.setStyleSheet(
                    "padding: 10px; font-size: 16px; background-color: #333; color: white; border: 2px solid #28a745;")
            else:
                self.bulk_input.setStyleSheet(
                    "padding: 10px; font-size: 16px; background-color: #333; color: white; border: 2px solid #555;")

        self.bulk_input.textChanged.connect(validate_bulk_input)

        btn_confirm.clicked.connect(dialog.accept)
        dialog_layout.addWidget(btn_confirm)

        if dialog.exec_() == QDialog.Accepted:
            self.bulk_name = self.bulk_input.text().strip()
            self.final_selection = [fid for fid, chk in check_boxes.items() if chk.isChecked()]
            return self.final_selection, self.bulk_name

        return None, None

    def show_startup_progress(self, total):
        self.progress_dialog = QProgressDialog("Analizowanie bazy...", "Anuluj", 0, total)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.show()

    def update_startup_progress(self, value, text):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)
            QApplication.processEvents()

    def close_startup_progress(self):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

    # Wewnątrz klasy okna GUI
    def start_scanning(self):
        def update_bar(current, total, text):
            self.progressBar.setMaximum(total)
            self.progressBar.setValue(current)
            self.statusLabel.setText(text)
            # Wymuszenie odświeżenia GUI, jeśli nie używasz wątków
            self.app.processEvents()

        self.controller.run_initial_scan(progress_callback=update_bar)

    def update_progress(self, current: int, total: int, message: str = ""):
        """
        Aktualizuje pasek postępu oraz (opcjonalnie) etykietę statusu.
        current: aktualny numer elementu
        total: całkowita liczba elementów
        message: tekst wyświetlany na pasku lub statusie
        """
        if total > 0:
            percentage = int((current / total) * 100)
            self.progressBar.setValue(percentage)

            # Opcjonalnie: ustawienie tekstu na pasku (jeśli progressBar.setTextVisible(True))
            if message:
                self.progressBar.setFormat(f"{message} ({percentage}%)")
            else:
                self.progressBar.setFormat(f"%p%")

        # Bardzo ważne: Wymuszenie przerysowania widżetu
        self.progressBar.repaint()

    def update_face_stats(self, count: int):
        """Aktualizuje tekst statystyk na dolnym pasku."""
        self.lbl_stats.setText(f"Wykryto łącznie twarzy: {count}")

    @staticmethod
    def confirm_all_labels():
        reply = QMessageBox.question(
            None,  # Lub self.ui jeśli main.py dziedziczy/ma dostęp do QMainWindow
            'Generowanie wizualizacji',
            'Klasyfikacja zakończona pomyślnie. Czy chcesz wygenerować folder z podpisanymi zdjęciami (dla wszystkich twarzy w bazie)?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            return 0
        return 1