import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QScrollArea, QLineEdit, QFrame, QDialog, QGridLayout,
                             QProgressDialog, QCheckBox)
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

        # Dolny pasek kontrolny
        controls = QHBoxLayout()
        btn_refresh = QPushButton("Odśwież widok")
        btn_refresh.clicked.connect(lambda: print("Refreshing..."))
        controls.addWidget(btn_refresh)

        self.lbl_stats = QLabel("Załadowane twarze: 0")
        controls.addStretch()
        controls.addWidget(self.lbl_stats)
        layout.addLayout(controls)

    def refresh_classified_faces(self, face_data_list, callback):
        """
        Czyści siatkę i ładuje nowe twarze sklasyfikowane przez SVM.
        face_data_list: lista krotek (face_id, name)
        callback: funkcja w main.py obsługująca zmianę imienia
        """
        # Czyszczenie siatki
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        for i, (fid, name) in enumerate(face_data_list):
            card = FaceCard(fid, name)
            card.confirmed.connect(callback)
            self.grid_layout.addWidget(card, i // 6, i % 6)

        self.lbl_stats.setText(f"Załadowane twarze: {len(face_data_list)}")

    def bulk_verify_faces(self, face_ids):
        """ Pozostawiamy bez zmian - wyskakujące okno dla DBSCAN """
        self.final_selection = []
        self.bulk_name = ""
        dialog = QDialog(self)
        dialog.setWindowTitle("Wykryto nową grupę (DBSCAN / Mean Shift)")
        dialog.setMinimumSize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        self.bulk_input = QLineEdit()
        self.bulk_input.setPlaceholderText("Wpisz imię dla tej grupy...")
        self.bulk_input.setStyleSheet("padding: 10px; font-size: 16px; background-color: #333; color: white;")
        dialog_layout.addWidget(self.bulk_input)

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
                img_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio))

            chk = QCheckBox("To ta osoba")
            chk.setChecked(True)
            container.addWidget(img_label, alignment=Qt.AlignCenter)
            container.addWidget(chk, alignment=Qt.AlignCenter)

            frame = QFrame()
            frame.setLayout(container)
            inner_grid.addWidget(frame, i // 4, i % 4)
            check_boxes[fid] = chk

        scroll.setWidget(grid_widget)
        dialog_layout.addWidget(scroll)

        btn_confirm = QPushButton("Zatwierdź grupę")
        btn_confirm.setStyleSheet("background-color: #28a745; height: 40px; font-weight: bold;")
        btn_confirm.clicked.connect(lambda: dialog.accept())
        dialog_layout.addWidget(btn_confirm)

        if dialog.exec_() == QDialog.Accepted:
            self.bulk_name = self.bulk_input.text().strip()
            self.final_selection = [fid for fid, chk in check_boxes.items() if chk.isChecked()]
            return self.final_selection, self.bulk_name
        return None, None

    # --- Paski postępu pozostają bez zmian ---
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