import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow, 
QWidget, QDoubleSpinBox, QVBoxLayout, QLineEdit, QHBoxLayout, QPushButton,
QPlainTextEdit, QSpinBox)

from solvers import SystemSolver
import backend


def create_hint(string):
        """
        Standart hint for user 
        """
        hint = QLabel(string)
        hint.setMaximumHeight(26)
        hint.setFrameStyle(0)
        return hint

    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.is_added = False
        hint = create_hint("Введіть значення параметрів: ")

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(hint)
        self.main_layout.addWidget(self.create_field('T:'))
        self.main_layout.addWidget(self.create_field("l_0:"))
        self.main_layout.addWidget(self.create_field("l_g1:"))
        self.main_layout.addWidget(self.create_field("l_g2:"))
        self.main_layout.addWidget(self.create_field("m_0:"))
        self.main_layout.addWidget(self.create_field("m_gamma:"))
        self.main_layout.addWidget(self.create_field("a0:"))
        self.main_layout.addWidget(self.create_field("b0:"))
        self.main_layout.addWidget(self.create_field("a1:"))
        self.main_layout.addWidget(self.create_field("b1:"))

        self.button_pass_args = QPushButton()
        self.button_pass_args.setText("Ввести")
        self.button_pass_args.clicked.connect(self.input_button_clicked)

        self.main_layout.addWidget(self.button_pass_args)
        
        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)

        self.setCentralWidget(main_widget) 
    
    def create_field(self, field_title: str, values_num: int=1):
        hint = create_hint(field_title)
        second_input = QHBoxLayout()
        for _ in range(values_num):
            value_input = QSpinBox()
            value_input.setMaximum(500)
            second_input.addWidget(hint)
            second_input.addWidget(value_input)

        second_input_widget = QWidget()
        second_input_widget.setLayout(second_input)

        return second_input_widget

    def input_button_clicked(self):
        """ 
        Slot for processing signal from
        first button, that adds input
        matrix and vector of right dimension
        to the main layout.
        """
        if self.is_added:
            self.delete_from_main_layout() # if widgets already added - remove them
            self.is_added = False
        else:
            self.is_added = True
            values = self.get_data_from_widgets()
            args = self.set_dict(values)
            lab4.arg = args
            lab4.main()

    def set_dict(self, values):
        arg = {
            "T": values[0],
            "l_0": values[1],
            "l_g": [values[2], values[3]],
            "m_0": values[4],
            "m_gamma": values[5],
            "a0": values[6],
            "b0": values[7],
            "a1": values[8],
            "b1": values[8],
        }
        return arg

    def get_data_from_widgets(self):
        items, values = [], [] 
        for i in range(1, self.main_layout.count() - 1):
            items.append(self.main_layout.itemAt(i).widget())
        for widget in items:
            child_widgets = widget.children()
            if isinstance(child_widgets, list):
                values.append(int(child_widgets[-1].cleanText()))
        return values
          
    def delete_from_main_layout(self):
        """
        Clearing layout from widgets
        after updating m and n
        """
        items = []
        for i in range(1, self.main_layout.count() - 1):
            items.append(self.main_layout.itemAt(i).widget())
        for widget in items:
            widgets = widget.children()
            if isinstance(widgets, list):
                widgets[-1].setValue(0)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
