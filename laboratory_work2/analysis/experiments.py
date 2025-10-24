from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from .generation import make_p_x, make_p_x_given_y, forward_from_doc
from .entropy import information_incomplete_reliability
from .validate import quick_checks, diagnostics_block


@dataclass
class RunStats:
    h_x: float
    h_x_given_y: float
    i_xy: float


def _fmt_vec(v: np.ndarray, ndigits: int = 6) -> str:
    return " ".join(f"{x:.{ndigits}f}" for x in v.tolist())


def _preview_matrix(a: np.ndarray, k: int = 5, ndigits: int = 5) -> str:
    r, c = a.shape
    rr, cc = min(k, r), min(k, c)
    lines = []
    for i in range(rr):
        row = " ".join(f"{a[i, j]:.{ndigits}f}" for j in range(cc))
        lines.append(row + (" ..." if cc < c else ""))
    return "\n".join(lines) + ("\n..." if rr < r else "")


def _save_csv(dirpath: Path, name: str, arr: np.ndarray | list[float]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"{name}.csv"
    np.savetxt(p, np.asarray(arr, dtype=float), delimiter=",", fmt="%.10f")
    print(f"[Сохр] Файл сохранён: {p}")


def run_experiments(
    N: int,
    experiments: int,
    rng: np.random.Generator,
    out_dir: Path,
    min_correct: float = 0.70,
) -> List[RunStats]:
    """
    Полный цикл ЛР-2 (логирование на русском):
      a) P(X)               — (2.2)
      b) P(X|Y)             — (2.3), diag ≥ 0.70
      c) P(Y)               — (2.4)
      d) P(X,Y)             — (2.6),(2.7)
      e) H(X)               — (2.1)
      f) H(X|Y)             — (2.8)
      g) I(X,Y)=H(X)-H(X|Y) — (2.9)
    + инварианты и сохранение CSV.
    """
    if experiments < 6:
        experiments = 6

    print(f"[Иниц] N = {N}, экспериментов = {experiments}, min_correct = {min_correct:.2f}")
    stats: List[RunStats] = []

    for k in range(1, experiments + 1):
        print(f"\n=== Эксперимент {k} ===")

        # (a) P(X)
        p_x = make_p_x(N, rng)
        print(f"[Ген  ] P(X) длина={N}, сумма Σ= {p_x.sum():.12f} -> {'ОК' if np.isclose(p_x.sum(), 1.0) else 'ВНИМАНИЕ'}")
        print(f"[Ген  ] P(X): {_fmt_vec(p_x, 6)}")

        # (b) P(X|Y)
        p_x_given_y = make_p_x_given_y(N, rng, min_correct=min_correct)
        col_sums = p_x_given_y.sum(axis=0)
        diag = np.diag(p_x_given_y)
        print(f"[Ген  ] P(X|Y) размер = {p_x_given_y.shape}; суммы по столбцам (первые 6): {_fmt_vec(col_sums[:6], 6)}")
        print(f"[Ген  ] диагональ p(x_j|y_j): минимум = {diag.min():.6f}; максимум = {diag.max():.6f}")
        print("[Ген  ] Превью P(X|Y) (5×5):\n" + _preview_matrix(p_x_given_y, 5, 5))

        # (c,d) P(Y), P(X,Y) по методичке
        p_y, p_xy = forward_from_doc(p_x, p_x_given_y)
        print(f"[Расч ] P(Y) длина={N}, сумма Σ= {p_y.sum():.12f} -> {'ОК' if np.isclose(p_y.sum(), 1.0) else 'ВНИМАНИЕ'}")
        print(f"[Расч ] P(Y): {_fmt_vec(p_y, 6)}")
        print(f"[Расч ] P(X,Y) размер = {p_xy.shape}, сумма Σ= {p_xy.sum():.12f} -> "
              f"{'ОК' if np.isclose(p_xy.sum(), 1.0) else 'ВНИМАНИЕ'}")
        print(f"[Пров ] Проверка Σ_i P(X,Y)[i,j] == P(Y)[j]: "
              f"{'ОК' if np.allclose(p_xy.sum(axis=0), p_y, atol=1e-12) else 'ВНИМАНИЕ'}")
        print("[Расч ] Превью P(X,Y) (5×5):\n" + _preview_matrix(p_xy, 5, 5))

        # (e,f,g) H(X), H(X|Y), I(X,Y)
        h_x, h_x_given_y, i_xy = information_incomplete_reliability(p_x, p_xy, p_x_given_y)
        print(f"[Рез  ] H(X)      = {h_x:.6f} бит")
        print(f"[Рез  ] H(X|Y)    = {h_x_given_y:.6f} бит")
        print(f"[Рез  ] I(X,Y)    = {i_xy:.6f} бит")
        print(f"[Пров ] Неотрицательность: "
              f"{'ОК' if (h_x >= -1e-12 and h_x_given_y >= -1e-12 and i_xy >= -1e-12) else 'ВНИМАНИЕ'}")

        # Инварианты и диагностика
        quick_checks(p_x, p_x_given_y, p_y, p_xy, min_correct=min_correct)
        diagnostics_block(p_x, p_y, p_xy)

        # CSV
        exp_dir = out_dir / f"exp_{k:02d}"
        _save_csv(exp_dir, "P_X", p_x)
        _save_csv(exp_dir, "P_Y", p_y)
        _save_csv(exp_dir, "P_X_given_Y", p_x_given_y)
        _save_csv(exp_dir, "P_XY", p_xy)

        stats.append(RunStats(h_x=h_x, h_x_given_y=h_x_given_y, i_xy=i_xy))

    # Сводка
    i_vals = np.array([s.i_xy for s in stats], dtype=float)
    print("\n=== Итоги серии ===")
    print(f"[Стат] Среднее Ī(X,Y) = {i_vals.mean():.6f} бит")
    print(f"[Стат] Мин/Макс I(X,Y) = {i_vals.min():.6f} / {i_vals.max():.6f} бит")
    return stats
