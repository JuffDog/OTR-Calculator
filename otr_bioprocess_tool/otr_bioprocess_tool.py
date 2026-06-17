import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QGridLayout, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class OTRModel:
    def __init__(self):
        self.A = 180
        self.alpha = 1.25
        self.beta = 0.85
        self.gamma = -0.65
        self.delta = -0.4
        self.k_osmo = 0.35

    def kla(self, n, d0, VL, d, osmo):
        kla = (
            self.A *
            (n ** self.alpha) *
            (d0 ** self.beta) *
            (VL ** self.gamma) *
            (d ** self.delta)
        )
        return kla * np.exp(-self.k_osmo * (osmo - 0.3))

    def predict(self, d, VL, d0, osmo, p, o2, rpm):
        d = d / 1000
        d0 = d0 / 1000
        VL = VL / 1e6
        n = rpm / 60

        kla = self.kla(n, d0, VL, d, osmo)
        return kla * (p * o2 / 0.21), kla

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.model = OTRModel()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("OTR Bioprocess Design Tool")
        layout = QVBoxLayout()
        grid = QGridLayout()

        self.inputs = {}
        labels = [
            ("Flask diameter [mm]", 80),
            ("Filling volume [ml]", 50),
            ("Shaking diameter [mm]", 25),
            ("Osmolality [osmol/kg]", 0.3),
            ("Pressure [bar]", 1.0),
            ("O2 fraction", 0.21),
            ("RPM", 250),
            ("OUR [mmol/L/h]", 50)
        ]

        for i, (text, default) in enumerate(labels):
            label = QLabel(text)
            spin = QDoubleSpinBox()
            spin.setRange(0, 10000)
            spin.setValue(default)
            self.inputs[text] = spin

            grid.addWidget(label, i, 0)
            grid.addWidget(spin, i, 1)

        layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self.calculate)

        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export)

        btn_layout.addWidget(calc_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

        self.result_label = QLabel("Results will appear here")
        layout.addWidget(self.result_label)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def calculate(self):
        d = self.inputs["Flask diameter [mm]"].value()
        VL = self.inputs["Filling volume [ml]"].value()
        d0 = self.inputs["Shaking diameter [mm]"].value()
        osmo = self.inputs["Osmolality [osmol/kg]"].value()
        p = self.inputs["Pressure [bar]"].value()
        o2 = self.inputs["O2 fraction"].value()
        rpm = self.inputs["RPM"].value()
        our = self.inputs["OUR [mmol/L/h]"].value()

        otr, kla = self.model.predict(d, VL, d0, osmo, p, o2, rpm)

        limitation = "OK"
        if our > otr:
            limitation = "OXYGEN LIMITED"

        self.result_label.setText(
            f"OTRmax: {otr:.2f} mmol/L/h\n"
            f"kLa: {kla:.2f} 1/h\n"
            f"OUR: {our:.2f}\n"
            f"Status: {limitation}"
        )

        self.plot(d, VL, d0, osmo, p, o2)

    def plot(self, d, VL, d0, osmo, p, o2):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        rpm_range = np.linspace(50, 400, 50)
        otr_vals = [
            self.model.predict(d, VL, d0, osmo, p, o2, r)[0]
            for r in rpm_range
        ]

        ax.plot(rpm_range, otr_vals)
        ax.set_xlabel("RPM")
        ax.set_ylabel("OTRmax")

        self.canvas.draw()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            data = {key: self.inputs[key].value() for key in self.inputs}
            df = pd.DataFrame([data])
            df.to_csv(path, index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
