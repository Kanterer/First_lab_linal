from __future__ import annotations

import csv
import os
from timeit import default_timer as timer
from typing import Callable

import numpy as np

from gauss import gauss_no_pivot, gauss_partial_pivot, solve_many_by_gauss_pivot
from lu import (
    EPS,
    lu_decomposition,
    solve_lu,
    solve_lu_from_decomposition,
    solve_lu_multiple_rhs,
)

RESULTS_DIR = "results"


# Генерация матриц

def generate_random_matrix(
    n: int,
    seed: int,
    make_diagonally_dominant: bool = True,
) -> np.ndarray:
    """
    Генерирует случайную матрицу.

    По умолчанию добавляется диагональное доминирование, чтобы методы без выбора
    ведущего элемента и LU без перестановок не падали на случайных почти нулевых pivot.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))

    if make_diagonally_dominant:
        A += n * np.eye(n)

    return A


def generate_random_vector(n: int, seed: int) -> np.ndarray:
    """Генерирует случайный вектор размера n."""
    rng = np.random.default_rng(seed)

    return rng.uniform(-1.0, 1.0, size=n)


def generate_random_rhs_matrix(n: int, k: int, seed: int) -> np.ndarray:
    """Генерирует матрицу правых частей размера n x k."""
    rng = np.random.default_rng(seed)

    return rng.uniform(-1.0, 1.0, size=(n, k))


def hilbert_matrix(n: int) -> np.ndarray:
    """Матрица Гильберта: H[i,j] = 1 / (i + j - 1)."""
    i = np.arange(1, n + 1)
    j = np.arange(1, n + 1)

    return 1.0 / (i[:, None] + j[None, :] - 1.0)


# Метрики точности

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Невязка ||Ax - b||."""
    return float(np.linalg.norm(A @ x - b))


def relative_residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Относительная невязка ||Ax - b|| / ||b||."""
    denom = np.linalg.norm(b)

    if denom < EPS:
        return residual_norm(A, x, b)

    return float(np.linalg.norm(A @ x - b) / denom)


def relative_error_norm(x_calculated: np.ndarray, x_true: np.ndarray) -> float:
    """Относительная ошибка ||x_calc - x_true|| / ||x_true||."""
    denom = np.linalg.norm(x_true)

    if denom < EPS:
        return float(np.linalg.norm(x_calculated - x_true))

    return float(np.linalg.norm(x_calculated - x_true) / denom)


# Измерение времени

def measure_time(func: Callable[[], object], repeats: int = 3) -> float:
    """Возвращает лучшее время выполнения из repeats запусков."""
    best_time = float("inf")

    for _ in range(repeats):
        start = timer()
        func()
        end = timer()

        best_time = min(best_time, end - start)

    return best_time


# Эксперимент 1: сравнение времени решения одной системы

def experiment_single_system(
    sizes: list[int] | None = None,
    repeats: int = 3,
) -> list[dict]:
    """Сравнивает время решения одной системы Ax = b разными методами."""
    if sizes is None:
        sizes = [100, 200, 500, 1000]

    rows = []

    print("\n=== Эксперимент 1: время решения одной системы ===")

    for n in sizes:
        print(f"n = {n}")

        A = generate_random_matrix(n, seed=1000 + n)
        b = generate_random_vector(n, seed=2000 + n)

        t_gauss_no_pivot = measure_time(
            lambda: gauss_no_pivot(A, b),
            repeats=repeats,
        )
        x_gauss_no_pivot = gauss_no_pivot(A, b)

        t_gauss_pivot = measure_time(
            lambda: gauss_partial_pivot(A, b),
            repeats=repeats,
        )
        x_gauss_pivot = gauss_partial_pivot(A, b)

        t_lu_decomposition = measure_time(
            lambda: lu_decomposition(A),
            repeats=repeats,
        )

        L, U = lu_decomposition(A)

        t_lu_solve = measure_time(
            lambda: solve_lu_from_decomposition(L, U, b),
            repeats=repeats,
        )
        x_lu = solve_lu_from_decomposition(L, U, b)

        row = {
            "n": n,
            "gauss_no_pivot_sec": t_gauss_no_pivot,
            "gauss_pivot_sec": t_gauss_pivot,
            "lu_decomposition_sec": t_lu_decomposition,
            "lu_solve_sec": t_lu_solve,
            "lu_total_sec": t_lu_decomposition + t_lu_solve,
            "residual_gauss_no_pivot": residual_norm(A, x_gauss_no_pivot, b),
            "residual_gauss_pivot": residual_norm(A, x_gauss_pivot, b),
            "residual_lu": residual_norm(A, x_lu, b),
        }

        rows.append(row)

    return rows


# Эксперимент 2: несколько правых частей

def experiment_multiple_rhs(
    n: int = 500,
    rhs_counts: list[int] | None = None,
    repeats: int = 3,
) -> list[dict]:
    """
    Сравнивает решение нескольких систем с одной матрицей A,
    но разными правыми частями b_1, ..., b_k.
    """
    if rhs_counts is None:
        rhs_counts = [1, 10, 100]

    rows = []

    print("\n=== Эксперимент 2: несколько правых частей ===")
    print(f"Фиксированный размер матрицы n = {n}")

    A = generate_random_matrix(n, seed=3000)

    for k in rhs_counts:
        print(f"k = {k}")

        B = generate_random_rhs_matrix(n, k, seed=4000 + k)

        local_repeats = repeats if k <= 10 else 1

        t_gauss_total = measure_time(
            lambda: solve_many_by_gauss_pivot(A, B),
            repeats=local_repeats,
        )
        X_gauss = solve_many_by_gauss_pivot(A, B)

        t_lu_decomposition = measure_time(
            lambda: lu_decomposition(A),
            repeats=repeats,
        )

        L, U = lu_decomposition(A)

        t_lu_solve_all = measure_time(
            lambda: solve_lu_multiple_rhs(L, U, B),
            repeats=local_repeats,
        )
        X_lu = solve_lu_multiple_rhs(L, U, B)

        row = {
            "n": n,
            "k": k,
            "gauss_pivot_total_sec": t_gauss_total,
            "lu_decomposition_sec": t_lu_decomposition,
            "lu_solve_all_rhs_sec": t_lu_solve_all,
            "lu_total_sec": t_lu_decomposition + t_lu_solve_all,
            "relative_residual_gauss": float(
                np.linalg.norm(A @ X_gauss - B) / np.linalg.norm(B)
            ),
            "relative_residual_lu": float(
                np.linalg.norm(A @ X_lu - B) / np.linalg.norm(B)
            ),
        }

        rows.append(row)

    return rows


# Эксперимент 3: точность на матрицах Гильберта

def experiment_hilbert_accuracy(
    sizes: list[int] | None = None,
) -> list[dict]:
    """Проверяет точность методов на плохо обусловленных матрицах Гильберта."""
    if sizes is None:
        sizes = [5, 10, 15]

    rows = []

    print("\n=== Эксперимент 3: точность на матрицах Гильберта ===")

    for n in sizes:
        print(f"n = {n}")

        H = hilbert_matrix(n)
        x_true = np.ones(n)
        b = H @ x_true

        methods = {
            "gauss_no_pivot": gauss_no_pivot,
            "gauss_pivot": gauss_partial_pivot,
            "lu": solve_lu,
        }

        for method_name, method in methods.items():
            try:
                x_calc = method(H, b)

                row = {
                    "n": n,
                    "method": method_name,
                    "relative_error": relative_error_norm(x_calc, x_true),
                    "residual": residual_norm(H, x_calc, b),
                    "relative_residual": relative_residual_norm(H, x_calc, b),
                }

            except Exception as error:
                row = {
                    "n": n,
                    "method": method_name,
                    "relative_error": None,
                    "residual": None,
                    "relative_residual": None,
                    "error": str(error),
                }

            rows.append(row)

    return rows


# Сохранение и вывод результатов

def save_csv(filename: str, rows: list[dict]) -> None:
    """Сохраняет результаты эксперимента в CSV."""
    if not rows:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    fieldnames = list(rows[0].keys())

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV сохранён: {path}")


def format_value(value: object) -> str:
    """Форматирует значение для красивого вывода в консоль."""
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def print_table(title: str, rows: list[dict]) -> None:
    """Печатает таблицу результатов в консоль."""
    if not rows:
        print(f"\n{title}: нет данных")
        return

    print(f"\n{title}")

    columns = list(rows[0].keys())

    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    widths = {}

    for column in columns:
        max_value_width = max(len(format_value(row.get(column))) for row in rows)
        widths[column] = max(len(column), max_value_width)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)

    for row in rows:
        print(
            " | ".join(
                format_value(row.get(column)).ljust(widths[column])
                for column in columns
            )
        )


# Графики

def plot_single_system(rows: list[dict]) -> None:
    """Строит график времени решения одной системы."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график experiment_single_system не построен")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_values = [row["n"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(
        n_values,
        [row["gauss_no_pivot_sec"] for row in rows],
        marker="o",
        label="Гаусс без выбора",
    )
    plt.plot(
        n_values,
        [row["gauss_pivot_sec"] for row in rows],
        marker="o",
        label="Гаусс с выбором",
    )
    plt.plot(
        n_values,
        [row["lu_total_sec"] for row in rows],
        marker="o",
        label="LU: разложение + решение",
    )

    plt.xlabel("Размер матрицы n")
    plt.ylabel("Время, секунды")
    plt.title("Сравнение времени решения одной СЛАУ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "single_system_time.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"График сохранён: {path}")


def plot_multiple_rhs(rows: list[dict]) -> None:
    """Строит график времени решения систем с несколькими правыми частями."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график experiment_multiple_rhs не построен")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    k_values = [row["k"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(
        k_values,
        [row["gauss_pivot_total_sec"] for row in rows],
        marker="o",
        label="Гаусс с выбором",
    )
    plt.plot(
        k_values,
        [row["lu_total_sec"] for row in rows],
        marker="o",
        label="LU: одно разложение + k решений",
    )

    plt.xlabel("Количество правых частей k")
    plt.ylabel("Время, секунды")
    plt.title("Экономия времени LU при нескольких правых частях")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "multiple_rhs_time.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"График сохранён: {path}")


def plot_hilbert_accuracy(rows: list[dict]) -> None:
    """Строит график точности методов на матрицах Гильберта."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график experiment_hilbert_accuracy не построен")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    methods = sorted(set(row["method"] for row in rows))

    plt.figure(figsize=(9, 5))

    for method in methods:
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row.get("relative_error") is not None
        ]

        n_values = [row["n"] for row in method_rows]
        errors = [row["relative_error"] for row in method_rows]

        plt.plot(n_values, errors, marker="o", label=method)

    plt.yscale("log")
    plt.xlabel("Размер матрицы Гильберта n")
    plt.ylabel("Относительная ошибка")
    plt.title("Точность методов на матрицах Гильберта")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "hilbert_accuracy.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"График сохранён: {path}")
