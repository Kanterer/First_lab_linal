from __future__ import annotations

import numpy as np

from lu import EPS, backward_substitution


def gauss_no_pivot(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает Ax = b методом Гаусса без выбора ведущего элемента."""
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()

    n = A.shape[0]

    for k in range(n - 1):
        pivot = A[k, k]

        if abs(pivot) < EPS:
            raise ZeroDivisionError(
                f"Слишком малый ведущий элемент A[{k},{k}] = {pivot}"
            )

        factors = A[k + 1:, k] / pivot

        A[k + 1:, k:] -= factors[:, None] * A[k, k:]
        b[k + 1:] -= factors * b[k]

        A[k + 1:, k] = 0.0

    return backward_substitution(A, b)


def gauss_partial_pivot(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решает Ax = b методом Гаусса с частичным выбором главного элемента.
    На каждом шаге выбирается максимальный по модулю элемент в текущем столбце.
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()

    n = A.shape[0]

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(A[k:, k]))

        if abs(A[pivot_row, k]) < EPS:
            raise ZeroDivisionError(f"Матрица вырождена на шаге {k}")

        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        pivot = A[k, k]
        factors = A[k + 1:, k] / pivot

        A[k + 1:, k:] -= factors[:, None] * A[k, k:]
        b[k + 1:] -= factors * b[k]

        A[k + 1:, k] = 0.0

    return backward_substitution(A, b)


def solve_many_by_gauss_pivot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Решает k систем Ax = b_j методом Гаусса с выбором.
    Каждая система решается заново.
    """
    n, k = B.shape
    X = np.zeros((n, k))

    for j in range(k):
        X[:, j] = gauss_partial_pivot(A, B[:, j])

    return X
