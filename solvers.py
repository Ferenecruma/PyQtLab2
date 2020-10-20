import numpy as np
from scipy.integrate import trapz
from scipy.linalg import det
from math import *


class SystemSolver():
    def __init__(self, matrix, vector, T):
        self.matrix = matrix
        self.vector = vector
        self.T = T
        self.n = len(matrix[0])

    def solve(self):
        # Transform matrix entries from str to functions
        self.__prepare_input(self.matrix)
        self.__prepare_input(self.vector)

        # Finding transpose of B
        matrix_transpose = list(map(list, zip(*self.matrix)))

        # Creating functions for integration 
        integral1 = self.__create_lambda(matrix_transpose, self.matrix)
        integral2 = self.__create_lambda(matrix_transpose, self.vector)
        
        # Calculating matrix integrals
        start, end = 0, self.T 
        P_2 = self.__calculate_integral(integral1, start, end)
        B_b = self.__calculate_integral(integral2, start, end)

        try:
            inv_A = np.linalg.pinv(P_2)

            def param_funcion(v=np.zeros((self.n, 1))):
                return inv_A.dot(B_b) + v - inv_A.dot(P_2.dot(v))
            
            # Перевіряємо на однозначність
            if det(P_2) > 0: 
                res = param_funcion()
                message = "Знайшовся точний розвязок"
                accur = None
            else:
                res = param_funcion
                message = "Не знайшовся точний розвязок.Наближений : "
                accur = self.compute_accuracy(B_b, inv_A, end=end)
                
        except np.linalg.LinAlgError:
            res = None
            accur = None
            message = "Не вдалося знайти жодного розвязку системи."

        return res, message, accur
    
    def __create_lambda(self, matrix1, matrix2):
        g = lambda x: np.array([[matrix1[i][j](x) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]) \
                .dot(np.array([[matrix2[i][j](x) for j in range(len(matrix2[0]))] for i in range(len(matrix2))]))
        return g

    def __calculate_integral(self, g, start, end):
        generator = (g(x) for x in np.linspace(start = start, stop = end, num = 120))
        return trapz(np.array(list(generator)), axis=0)
    
    def compute_accuracy(self, B_b, inv_A, start=0, end=5):
        vec_transpose = list(map(list, zip(*self.vector)))
        integral3 = self.__create_lambda(vec_transpose, self.vector)
        accur = self.__calculate_integral(integral3, start, end) - B_b.T @ inv_A @ B_b
        return accur
    
    def __prepare_input(self, matrix):
        """Construct lambda function from string expression with arg t."""
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = eval('lambda t: '+ matrix[i][j]) 