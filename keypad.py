# keypad.py
from PyQt5 import QtCore, QtWidgets


class KeypadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Wprowadź PIN"):
        super().__init__(parent)

        # Dialog modalny, pełnoekranowy overlay bez ramki
        self.setModal(True)
        self.setWindowTitle(title)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Dialog
        )
        self.setStyleSheet("background-color: rgba(0,0,0,210); color: white;")

        # tutaj trzymamy ostateczną wartość PIN, żeby nie dotykać QLineEdit po zamknięciu
        self._value = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # górny pasek z tytułem i X
        topbar = QtWidgets.QHBoxLayout()

        lbl = QtWidgets.QLabel(title)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("font-size:28px; font-weight:600; color:white;")

        btn_close = QtWidgets.QPushButton("X")
        btn_close.setFixedSize(48, 48)
        btn_close.setStyleSheet(
            "font-size:24px; font-weight:700; border-radius:12px; "
            "background:#550000; color:white;"
        )
        btn_close.clicked.connect(self.reject)

        topbar.addWidget(lbl, 1)
        topbar.addWidget(btn_close, 0, QtCore.Qt.AlignRight)
        layout.addLayout(topbar)

        # pole PIN
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
        self.edit.setFixedHeight(60)
        self.edit.setStyleSheet(
            "font-size:32px; padding:8px; border-radius:12px; "
            "background:#222; color:white;"
        )
        layout.addWidget(self.edit)

        # klawiatura numeryczna
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)

        btnstyle = (
            "font-size:26px; padding:16px; border-radius:16px; "
            "background:#333; color:white;"
        )

        keys = [
            ("1", 0, 0), ("2", 0, 1), ("3", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("7", 2, 0), ("8", 2, 1), ("9", 2, 2),
            ("←", 3, 0), ("0", 3, 1), ("OK", 3, 2),
        ]

        for text, row, col in keys:
            btn = QtWidgets.QPushButton(text)
            btn.setStyleSheet(btnstyle)
            # x=text w domknięciu, żeby uniknąć problemu z lambdą
            btn.clicked.connect(lambda _, x=text: self.on_btn(x))
            grid.addWidget(btn, row, col)

        layout.addLayout(grid)

        self.resize(460, 640)

    def on_btn(self, t: str):
        if t == "OK":
            self.accept()
        elif t == "←":
            self.edit.setText(self.edit.text()[:-1])
        else:
            self.edit.setText(self.edit.text() + t)

    def accept(self):
        # zapamiętaj PIN zanim Qt ewentualnie zacznie sprzątać widgety
        self._value = self.edit.text()
        super().accept()

    def reject(self):
        # przy anulowaniu czyścimy wartość
        self._value = ""
        super().reject()

    def value(self) -> str:
        # zwracamy zapamiętaną wartość, nie dotykamy QLineEdit po zamknięciu
        return self._value
