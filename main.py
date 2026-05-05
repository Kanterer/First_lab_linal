from __future__ import annotations

import os

from experiments import (
    RESULTS_DIR,
    experiment_single_system,
    experiment_multiple_rhs,
    experiment_hilbert_accuracy,
    print_table,
    save_csv,
    plot_single_system,
    plot_multiple_rhs,
    plot_hilbert_accuracy,
)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    single_rows = experiment_single_system(
        sizes=[100, 200, 500, 1000],
        repeats=3,
    )
    print_table("Эксперимент 1: время решения одной системы", single_rows)
    save_csv("time_single_system.csv", single_rows)
    plot_single_system(single_rows)

    multiple_rhs_rows = experiment_multiple_rhs(
        n=500,
        rhs_counts=[1, 10, 100],
        repeats=3,
    )
    print_table("Эксперимент 2: несколько правых частей", multiple_rhs_rows)
    save_csv("multiple_rhs.csv", multiple_rhs_rows)
    plot_multiple_rhs(multiple_rhs_rows)

    hilbert_rows = experiment_hilbert_accuracy(
        sizes=[5, 10, 15],
    )
    print_table("Эксперимент 3: матрицы Гильберта", hilbert_rows)
    save_csv("hilbert_accuracy.csv", hilbert_rows)
    plot_hilbert_accuracy(hilbert_rows)
    print("\nГотово. Все результаты сохранены в папку results/")


if __name__ == "__main__":
    main()
