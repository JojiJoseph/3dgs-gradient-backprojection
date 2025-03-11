from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QHBoxLayout, QLineEdit
from PyQt5 import QtCore

class SliderWithInput(QWidget):
    def __init__(self, name, min_val, max_val, default_val, callback):
        super().__init__()
        
        self.layout = QHBoxLayout(self)
        self.label = QLabel(name)
        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)

        self.input_field = QLineEdit(str(default_val))
        self.input_field.setFixedWidth(50)

        # Synchronize slider and input field
        self.slider.valueChanged.connect(self.update_input)
        self.input_field.textChanged.connect(self.update_slider)

        # Callback function
        self.callback = callback
        self.slider.valueChanged.connect(callback)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.input_field)

    def update_slider(self, val):
        if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
            int_val = int(val)
            if self.slider.minimum() <= int_val <= self.slider.maximum():
                self.slider.setValue(int_val)

    def update_input(self, val):
        self.input_field.setText(str(val))
