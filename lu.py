from __future__ import annotations

import numpy as np

EPS = 1e-12


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает Ly = b, где L — нижнетреугольная матрица."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)

    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        if abs(L[i, i]) < EPS:
            raise ZeroDivisionError(f"Нулевой диагональный элемент L[{i},{i}]")

        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Решает Ux = y, где U — верхнетреугольная матрица."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)

    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < EPS:
            raise ZeroDivisionError(f"Нулевой диагональный элемент U[{i},{i}]")

        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def forward_substitution_matrix(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Решает LY = B для нескольких правых частей."""
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)

    n, k = B.shape
    Y = np.zeros((n, k))

    for i in range(n):
        if abs(L[i, i]) < EPS:
            raise ZeroDivisionError(f"Нулевой диагональный элемент L[{i},{i}]")

        Y[i, :] = (B[i, :] - L[i, :i] @ Y[:i, :]) / L[i, i]

    return Y


def backward_substitution_matrix(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Решает UX = Y для нескольких правых частей."""
    U = np.asarray(U, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n, k = Y.shape
    X = np.zeros((n, k))

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < EPS:
            raise ZeroDivisionError(f"Нулевой диагональный элемент U[{i},{i}]")

        X[i, :] = (Y[i, :] - U[i, i + 1:] @ X[i + 1:, :]) / U[i, i]

    return X


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Выполняет LU-разложение A = LU без перестановок.
    L — нижнетреугольная матрица с единицами на диагонали.
    U — верхнетреугольная матрица.
    """
    U = np.asarray(A, dtype=float).copy()
    n = U.shape[0]

    L = np.eye(n)

    for k in range(n - 1):
        pivot = U[k, k]

        if abs(pivot) < EPS:
            raise ZeroDivisionError(
                f"LU без перестановок невозможно: U[{k},{k}] = {pivot}"
            )

        L[k + 1:, k] = U[k + 1:, k] / pivot

        U[k + 1:, k:] -= L[k + 1:, k][:, None] * U[k, k:]
        U[k + 1:, k] = 0.0

    if abs(U[-1, -1]) < EPS:
        raise ZeroDivisionError(
            f"LU без перестановок невозможно: U[-1,-1] = {U[-1, -1]}"
        )

    return L, U


def solve_lu_from_decomposition(
    L: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Решает Ax = b, если A уже разложена как A = LU."""
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Полное решение через LU: сначала разложение, потом две подстановки."""
    L, U = lu_decomposition(A)

    return solve_lu_from_decomposition(L, U, b)


def solve_lu_multiple_rhs(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Решает AX = B для нескольких правых частей при уже известном A = LU."""
    Y = forward_substitution_matrix(L, B)
    X = backward_substitution_matrix(U, Y)

    return X
