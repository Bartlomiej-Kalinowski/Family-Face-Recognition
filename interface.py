"""PyQt user interface components for face review and labeling workflows."""

import os
import sys

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config import Config


class FaceCard(QFrame):
    """Single face tile with editable label and confirm action."""

    confirmed = pyqtSignal(str, str)

    def __init__(self, face_id, name, parent=None):
        """Build a card for one detected face."""
        super().__init__(parent)
        self.face_id = face_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(160)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; border: 1px solid #3e3e3e;")

        layout = QVBoxLayout()

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        face_path = os.path.join(Config.FACES_DIR, f"{face_id}.jpg")
        pixmap = QPixmap(face_path)
        if not pixmap.isNull():
            self.lbl_img.setPixmap(pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(self.lbl_img)

        self.input_name = QLineEdit()
        self.input_name.setText(name)
        self.input_name.setStyleSheet(
            "padding: 5px; color: #ffffff; background-color: #1e1e1e; border: 1px solid #555; font-weight: bold;"
        )
        self.input_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.input_name)

        self.btn_confirm = QPushButton("Zmień / OK")
        self.btn_confirm.setStyleSheet("background-color: #444; color: white; padding: 4px;")
        self.btn_confirm.clicked.connect(self._on_confirm)
        layout.addWidget(self.btn_confirm)

        self.setLayout(layout)

    def _on_confirm(self):
        """Emit the edited label and visually mark the card as reviewed."""
        new_name = self.input_name.text().strip()
        self.confirmed.emit(self.face_id, new_name)
        self.setStyleSheet("background-color: #1a3320; border: 1px solid #28a745;")


class FaceInterface(QMainWindow):
    """Main application window used to verify and adjust classifications."""

    def __init__(self):
        """Create the application and initialize window widgets."""
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
        """Apply a consistent dark palette to all Qt widgets."""
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
        """Build the primary layout, grid area, and status controls."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        header = QLabel("WYNIKI AUTOMATYCZNEJ KLASYFIKACJI (SVM)")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #007acc; margin: 10px;")
        layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #1e1e1e;")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll)

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
        self.progressBar.setStyleSheet(
            """
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
            """
        )
        progress_layout.addWidget(self.progressBar)
        layout.addWidget(self.progress_container)

        controls = QHBoxLayout()
        btn_refresh = QPushButton("Odśwież widok")
        btn_refresh.clicked.connect(lambda: print("Refreshing..."))
        controls.addWidget(btn_refresh)

        self.btn_generate_visualization = QPushButton("Generuj wizualizacje")
        self.btn_generate_visualization.setEnabled(False)
        self.btn_generate_visualization.setVisible(False)
        self.btn_generate_visualization.setStyleSheet(
            """
            QPushButton:disabled { background-color: #444; color: #888; }
            QPushButton:enabled { background-color: #1f6f8b; color: white; font-weight: bold; }
            """
        )
        controls.addWidget(self.btn_generate_visualization)

        self.lbl_stats = QLabel("Załadowane twarze: 0")
        controls.addStretch()
        controls.addWidget(self.lbl_stats)
        layout.addLayout(controls)

    def ask_for_scan_mode(self):
        """Ask the user which data preparation mode should be used."""
        msg = QMessageBox()
        msg.setWindowTitle("Tryb Skanowania")
        msg.setText("Wybierz sposób przygotowania bazy:")
        full_btn = msg.addButton("Pełny Skan (YOLO)", QMessageBox.ActionRole)
        incr_btn = msg.addButton("Przyrostowy", QMessageBox.ActionRole)
        exist_btn = msg.addButton("Mam już wycięte twarze", QMessageBox.ActionRole)
        msg.addButton("Anuluj", QMessageBox.RejectRole)

        msg.exec_()

        if msg.clickedButton() == full_btn:
            return "full"
        if msg.clickedButton() == incr_btn:
            return "incremental"
        if msg.clickedButton() == exist_btn:
            return "use_existing"
        return "cancel"

    def ask_for_preprocessing_type(self, dataset: int = 1):
        """Ask the user which type of embeddings should be used."""
        msg = QMessageBox()
        msg.setWindowTitle("Wybor reprezentacji wektorow twarzy")
        msg.setText(f"Wybierz sposob przetwarzania zdjęć twarzy w zbiorze danych nr. {dataset}:")
        hog_embds = msg.addButton("HOG (Podstawowe)", QMessageBox.ActionRole)
        neural_network_cv2 = msg.addButton("Siec neuronowa (zaawanasowane)", QMessageBox.ActionRole) # neural network + face alignment
        msg.addButton("Anuluj", QMessageBox.RejectRole)

        msg.exec_()

        if msg.clickedButton() == hog_embds:
            return "hog"
        if msg.clickedButton() == neural_network_cv2:
            return "neural_network"
        return "cancel"

    def ask_for_scan_dataset_id(self, title, comment) -> int:
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(comment)
        fst_btn = msg.addButton("1", QMessageBox.ActionRole)
        snd_btn = msg.addButton("2", QMessageBox.ActionRole)
        trd_btn = msg.addButton("3", QMessageBox.ActionRole)
        msg.addButton("Anuluj", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == fst_btn:
            return 1
        elif msg.clickedButton() == snd_btn:
            return 2
        elif msg.clickedButton() == trd_btn:
            return 3
        else:
            return -1

    def refresh_classified_faces(self, face_data_list, callback):
        """Rebuild the grid from `(face_id, label, ...)` records."""
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        for i, row in enumerate(face_data_list):
            fid, name = self._parse_face_row(row)
            if not fid:
                continue
            card = FaceCard(fid, name)
            card.confirmed.connect(callback)
            self.grid_layout.addWidget(card, i // 6, i % 6)

        self.lbl_stats.setText(f"Załadowane twarze: {len(face_data_list)}")

    @staticmethod
    def _parse_face_row(row):
        """Normalize row formats from SVM output and full DB output."""
        if len(row) >= 3 and isinstance(row[0], str) and os.path.sep in row[0]:
            return str(row[1]), str(row[2] or "Unknown")

        if len(row) >= 2:
            return str(row[0]), str(row[1] or "Unknown")

        return "", "Unknown"

    def set_visualize_callback(self, callback):
        """Connect controller callback for visualization generation action."""
        self.btn_generate_visualization.clicked.connect(callback)

    def set_visualization_enabled(self, enabled: bool):
        """Show and toggle visualization button after SVM phase."""
        self.btn_generate_visualization.setVisible(True)
        self.btn_generate_visualization.setEnabled(enabled)

    def bulk_verify_faces(self, face_id_path_pairs: dict):
        """Open a dialog to assign one label to a selected cluster."""
        self.final_selection = []
        self.bulk_name = ""
        dialog = QDialog(self)
        dialog.setWindowTitle("Wykryto nową grupę (DBSCAN)")
        dialog.setMinimumSize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        self.bulk_input = QLineEdit()
        self.bulk_input.setPlaceholderText("Wpisz imię dla tej grupy (WYMAGANE)...")
        self.bulk_input.setStyleSheet(
            """
            padding: 10px;
            font-size: 16px;
            background-color: #333;
            color: white;
            border: 2px solid #555;
            """
        )
        dialog_layout.addWidget(self.bulk_input)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        inner_grid = QGridLayout(grid_widget)

        check_boxes = {}
        for i, fid_and_path in enumerate(face_id_path_pairs.items()):
            fid = fid_and_path[0]
            face_path = fid_and_path[1]
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

        btn_confirm = QPushButton("Zatwierdź grupę")
        btn_confirm.setEnabled(False)
        btn_confirm.setStyleSheet(
            """
            QPushButton:disabled { background-color: #444; color: #888; }
            QPushButton:enabled { background-color: #28a745; color: white; font-weight: bold; }
            height: 40px;
            """
        )

        def validate_bulk_input():
            """Enable confirmation only when the name field is non-empty."""
            is_valid = len(self.bulk_input.text().strip()) > 0
            btn_confirm.setEnabled(is_valid)
            if is_valid:
                self.bulk_input.setStyleSheet(
                    "padding: 10px; font-size: 16px; background-color: #333; color: white; border: 2px solid #28a745;"
                )
            else:
                self.bulk_input.setStyleSheet(
                    "padding: 10px; font-size: 16px; background-color: #333; color: white; border: 2px solid #555;"
                )

        self.bulk_input.textChanged.connect(validate_bulk_input)

        btn_confirm.clicked.connect(dialog.accept)
        dialog_layout.addWidget(btn_confirm)

        if dialog.exec_() == QDialog.Accepted:
            self.bulk_name = self.bulk_input.text().strip()
            self.final_selection = [fid for fid, chk in check_boxes.items() if chk.isChecked()]
            return self.final_selection, self.bulk_name

        return None, None

    def show_startup_progress(self, total):
        """Show modal progress dialog used during startup operations."""
        self.progress_dialog = QProgressDialog("Analizowanie bazy...", "Anuluj", 0, total)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.show()

    def update_startup_progress(self, value, text):
        """Update startup dialog progress and text if the dialog exists."""
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)
            QApplication.processEvents()

    def close_startup_progress(self):
        """Close startup progress dialog if it was created."""
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

    def start_scanning(self):
        """Run initial scanning and stream updates into the progress widgets."""

        def update_bar(current, total, text):
            self.progressBar.setMaximum(total)
            self.progressBar.setValue(current)
            self.statusLabel.setText(text)
            self.app.processEvents()

        self.controller.run_initial_scan(progress_callback=update_bar)

    def update_progress(self, current: int, total: int, message: str = ""):
        """Update main progress bar with percentage and optional status message."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progressBar.setValue(percentage)

            if message:
                self.progressBar.setFormat(f"{message} ({percentage}%)")
            else:
                self.progressBar.setFormat("%p%")

        # Force repaint to keep UI responsive during long synchronous loops.
        self.progressBar.repaint()

    def update_face_stats(self, count: int):
        """Refresh bottom-bar text with total detected face count."""
        self.lbl_stats.setText(f"Wykryto łącznie twarzy: {count}")

    @staticmethod
    def confirm_all_labels():
        """Ask whether to generate labeled visualization outputs."""
        reply = QMessageBox.question(
            None,
            "Generowanie wizualizacji",
            "Czy na pewno wygenerować folder z podpisanymi zdjęciami? "
            "Zostaną użyte aktualne etykiety po Twoich poprawkach.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            return 0
        return 1
