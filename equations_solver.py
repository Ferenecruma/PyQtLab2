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


def create_hint(string):
        """
        Standart hint for user 
        """
        hint = QLabel(string)
        hint.setMaximumHeight(26)
        hint.setFrameStyle(0)
        return hint


class ResultPlotter():
    def __init__(self, res):
        self.xs = res

    def plot2(self):
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.xs[:, 0], y=self.xs[:, 1], ax=ax)

    def plot3(self):
        ax = plt.gca(projection='3d')
        ax.scatter(self.xs[:, 0], self.xs[:, 1], self.xs[:, 2])


class ResultWindow(QWidget):
    """
    Window for displaying the results  
    """
    def __init__(self, result=None, message="", accur=None):
        super().__init__()

        self.setWindowTitle("Результати")

        self.main_layout = QVBoxLayout()
        hint = create_hint(message)
        self.main_layout.addWidget(hint)
        
        if result is not None:
            self.layout = QGridLayout()
            self.compute_res = None

            if callable(result):
                self.compute_res = result
                result = result()
            
            self.m = len(result)

            for i in range(len(result)):
                self.layout.addWidget(QLabel(str(result[i])), i, 0)
            
            matrix_display = QWidget()
            matrix_display.setLayout(self.layout)
            self.main_layout.addWidget(matrix_display)

            # Plotting results
            if callable(self.compute_res) and self.m <= 3:
                random_vecs = self.generate_random_solutions(self.m)
                self.plot_res(random_vecs)
            elif self.m <= 3:
                self.plot_res(result.T)

        if accur:
            hint = create_hint("Похибка")
            self.main_layout.addWidget(hint)
            self.main_layout.addWidget(QLabel(str(accur)))

        self.setLayout(self.main_layout)

    def plot_res(self, res):
        plotter = ResultPlotter(res)
        if self.m == 2:
            plotter.plot2()
        else:
            plotter.plot3()

        plt.show()
    
    def generate_random_solutions(self, n):
        xs = []
        for _ in range(50):
            v = np.random.rand(n)
            x = self.compute_res(v)
            xs.append(x)
        xs = np.vstack(xs)
        return xs

    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.is_added = False

        self.main_layout = QVBoxLayout()
        first_input = QHBoxLayout()
        
        hint = create_hint("Введіть m та n :")
        self.main_layout.addWidget(hint)

        self.m_input = QSpinBox()
        self.n_input = QSpinBox()
        
        first_input.addWidget(self.m_input)
        first_input.addWidget(self.n_input)

        firs_input_widget = QWidget()
        firs_input_widget.setLayout(first_input)

        self.main_layout.addWidget(firs_input_widget)

        hint1 = create_hint("Введіть T :")
        second_input = QHBoxLayout()
        self.T_input = QSpinBox()
        second_input.addWidget(hint1)
        second_input.addWidget(self.T_input)

        second_input_widget = QWidget()
        second_input_widget.setLayout(second_input)
        self.main_layout.addWidget(second_input_widget)

        self.button_pass_args = QPushButton()
        self.button_pass_args.setText("Ввести")
        self.button_pass_args.clicked.connect(self.input_button_clicked)

        self.main_layout.addWidget(self.button_pass_args)
        
        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)

        self.setCentralWidget(main_widget) 
    
    def input_button_clicked(self):
        """ 
        Slot for processing signal from
        first button, that adds input
        matrix and vector of right dimension
        to the main layout.
        """
        if self.is_added:
            self.delete_from_main_layout() # if widgets already added - remove them
        else:
            self.is_added = True

        self.m, self.n = int(self.m_input.cleanText()), int(self.n_input.cleanText())
        self.T = int(self.T_input.cleanText())

        self.matrix, self.matrix_layout = self.create_matrix_input(self.m, self.n)
        self.vector_b, self.vector_layout = self.create_matrix_input(self.m, 1)
        hint = create_hint("Введіть вектор b:")
        
        self.button_compute = QPushButton()
        self.button_compute.setText("Знайти розвязок")
        self.button_compute.clicked.connect(self.compute_result)

        self.main_layout.addWidget(create_hint('Введіть матрицю A:'))
        self.main_layout.addWidget(self.matrix)

        self.main_layout.addWidget(hint)
        self.main_layout.addWidget(self.vector_b)
        self.main_layout.addWidget(self.button_compute)

    def create_matrix_input(self, m, n):
        """
        Creating input matrix of 
        spinBoxes with (m, n) shape
        """
        layout = QGridLayout()

        for i in range(m):
            for j in range(n):
                line = QLineEdit()
                layout.addWidget(line, i, j)

        widget = QWidget()
        widget.setLayout(layout)
        return widget, layout

    def get_data_from_table(self, layout, m, n):
        data = [[0 for i in range(n)] for j in range(m)]
        for i in range(m):
            for j in range(n):
                item = layout.itemAtPosition(i, j).widget()
                data[i][j] = item.text()
        return data
    
    def compute_result(self):
        matrix = self.get_data_from_table(self.matrix_layout, self.m, self.n)
        vector_b = self.get_data_from_table(self.vector_layout, self.m, 1)

        solver = SystemSolver(matrix, vector_b, self.T)
        res, message, accur = solver.solve()

        self.result_w = ResultWindow(res, message, accur)
        self.result_w.show()
        
    def delete_from_main_layout(self):
        """
        Clearing layout from widgets
        after updating m and n
        """
        items = []
        for i in range(2, self.main_layout.count()):
            items.append(self.main_layout.itemAt(i).widget())
        for widget in items:
            widget.hide()
            self.main_layout.removeWidget(widget)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
