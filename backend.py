# Чергикало Денис група ОМ-4

from tkinter.ttk import *
from tkinter import *
import numpy as np
import ast
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sympy
from sympy import *
from sympy.functions.elementary.complexes import sign
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

t, x0, x1, t_, x0_, x1_ = symbols("t x0 x1 t_ x0_ x1_")

arg = {
    "T": 1.0,
    "l_0": 200,
    "l_g": [100, 100],
    "m_0": 50,
    "m_gamma": 50,
    "a0": 0,
    "a1": 0,
    "b0": 1,
    "b1": 1,
}


class Discret_System:
    def __init__(self, str_example_solve, X_L_0, X_L_gamma, X_M_0, X_M_gamma):
        self.k = 1

        self.l_0 = arg["l_0"]
        self.l_gamma = sum(arg["l_g"])
        self.l_g = arg["l_g"]

        self.m_0 = arg["m_0"]
        self.m_gamma = arg["m_gamma"]

        self.T = arg["T"]  # t на [0,T]
        self.a, self.b = [arg["a0"], arg["a1"]], [arg["b0"], arg["b1"]]  # xi на [ai,bi]

        self.L = lambda exp: diff(exp, t) - self.k * (
            diff(diff(exp, x0), x0) + diff(diff(exp, x1), x1)
        )
        self.L_0 = lambda exp: exp
        self.L_gamma = lambda exp: [exp, exp]  # для x0 - с края a0 и b0 соответственно

        self.G = Piecewise(
            (0, (t - t_) <= 0),
            (
                Max(
                    0,
                    ((4 * parse_expr("pi") * self.k * (t - t_)) ** parse_expr("-1/2"))
                    * exp(-((x0 - x0_) ** 2) / (4 * self.k * (t - t_))),
                ),
                True,
            ),
        )

        self.real_u, self.cond0, self.cond_gamma = self.create_conditions(
            parse_expr(str_example_solve)
        )

        self.G_0, self.G_gamma = self.create_L_G()

        A11 = np.array(
            [[self.G_0(x_l.tolist(), x_m.tolist()) for x_m in X_M_0] for x_l in X_L_0]
        )
        A12 = np.array(
            [
                [self.G_0(x_l.tolist(), x_m.tolist()) for x_m in X_M_gamma]
                for x_l in X_L_0
            ]
        )
        A21 = np.array(
            [
                [
                    [self.G_gamma[i](x_l.tolist(), x_m.tolist()) for x_m in X_M_0]
                    for x_l in X_L_gamma[i]
                ]
                for i in np.arange(len(self.G_gamma))
            ]
        )
        A21 = A21.reshape(A21.shape[0] * A21.shape[1], A21.shape[2])
        A22 = np.array(
            [
                [
                    [self.G_gamma[i](x_l.tolist(), x_m.tolist()) for x_m in X_M_gamma]
                    for x_l in X_L_gamma[i]
                ]
                for i in np.arange(len(self.G_gamma))
            ]
        )
        A22 = A22.reshape(A22.shape[0] * A22.shape[1], A22.shape[2])

        A = np.zeros((self.l_0 + self.l_gamma, self.m_0 + self.m_gamma))
        Y = np.zeros((self.l_0 + self.l_gamma,))

        A[: self.l_0, : self.m_0] = A11
        A[: self.l_0, self.m_0 :] = A12
        A[self.l_0 :, : self.m_0] = A21
        A[self.l_0 :, self.m_0 :] = A22

        Y[: self.l_0] = self.cond0(X_L_0[:, 0], X_L_0[:, 1], X_L_0[:, 2])

        gamma_list = []
        for i in np.arange(len(self.G_gamma)):
            gamma_list += list(
                self.cond_gamma[i](
                    X_L_gamma[i][:, 0], X_L_gamma[i][:, 1], X_L_gamma[i][:, 2]
                )
            )

        Y[self.l_0 :] = np.array(gamma_list)

        self.solution = np.linalg.lstsq(A, Y)[0]

        self.plot(X_L_0, X_L_gamma, X_M_0)

    def create_conditions(self, exp):
        return (
            lambdify((t, x0, x1), self.L(exp).evalf(), "numpy"),
            lambdify((t, x0, x1), self.L_0(exp).evalf(), "numpy"),
            [lambdify((t, x0, x1), e.evalf(), "numpy") for e in self.L_gamma(exp)],
        )

    def create_L_G(self):
        return lambdify(
            ([t, x0, x1], [t_, x0_, x1_]), self.L_0(self.G).simplify(), "numpy"
        ), [
            lambdify(([t, x0, x1], [t_, x0_, x1_]), e.simplify(), "numpy")
            for e in self.L_gamma(self.G)
        ]

    def plot(self, X_L_0, X_L_gamma, X_M_0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X_M_0[:, 0], X_M_0[:, 1], self.real_u(X_M_0[:, 0], X_M_0[:, 1], X_M_0[:, 2])
        )
        ax.scatter(X_M_0[:, 0], X_M_0[:, 1], self.solution[: self.m_0])
        ax.legend(["дійсні значення u", "Передбачення"])
        ax.set_xlabel("t")
        ax.set_ylabel("x0")
        ax.set_zlabel("u")
        plt.show()


def create_points(n):
    M = np.random.random_sample(
        (n, 3)
    )  # Getting array of random floats of the shape (n, 3) from interval [0, 1)
    # Setting our samples to be from intervals [0, T), [a0, b0), [a1, b1)
    M[:, 0] = arg["T"] * M[:, 0]
    M[:, 1] = arg["a0"] + M[:, 1] * (arg["b0"] - arg["a0"])
    M[:, 2] = arg["a1"] + M[:, 2] * (arg["b1"] - arg["a1"])
    return M


def main():
    X_L_0 = create_points(arg["l_0"])
    X_L_0[:, 0] = -0.1 * (X_L_0[:, 0] + 0.1)
    X_L_0[:, 1] = (arg["b0"] + arg["a0"]) / 2 + (
        X_L_0[:, 1] - (arg["b0"] + arg["a0"]) / 2
    ) * 1.2

    X_L_gamma1 = create_points(arg["l_g"][0])
    X_L_gamma1[:, 0] = arg["T"] / 2 + 1.2 * (X_L_gamma1[:, 0] - arg["T"] / 2)
    X_L_gamma1[:, 1] = arg["a0"] - (X_L_gamma1[:, 1] - arg["a0"] + 0.1)

    X_L_gamma2 = create_points(arg["l_g"][1])
    X_L_gamma2[:, 0] = arg["T"] / 2 + 1.2 * (X_L_gamma2[:, 0] - arg["T"] / 2)
    X_L_gamma2[:, 1] = arg["b0"] + (X_L_gamma2[:, 1] - arg["a0"] + 0.1)

    X_L_gamma = [X_L_gamma1, X_L_gamma2]

    X_M_0 = create_points(arg["m_0"])

    X_M_gamma = create_points(arg["m_gamma"])

    S = Discret_System("cos(x0) + sin(x1) + t", X_L_0, X_L_gamma, X_M_0, X_M_gamma)


if __name__ == "__main__":
    main()
