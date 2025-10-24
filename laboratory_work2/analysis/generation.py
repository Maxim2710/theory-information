from __future__ import annotations

from typing import Tuple
import numpy as np


def make_p_x(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    (2.2) P(X) = (p(x1),...,p(xN)).
    Генерируем Dirichlet(1,...,1): p_i > 0 и Σ p_i = 1.
    """
    return rng.dirichlet(np.ones(n), size=1)[0]


def make_p_x_given_y(
    n: int,
    rng: np.random.Generator,
    min_correct: float = 0.70,
) -> np.ndarray:
    """
    (2.3) Матрица условных вероятностей P(X|Y) размера N×N (столбцы суммируются к 1).
    Для каждого столбца j:
      • p(x_j | y_j) ~ U[min_correct, 1];
      • остаток равномерно по i ≠ j (через Dirichlet);
      • сумма по i в столбце = 1.
    """
    if not (0.0 <= min_correct <= 1.0):
        raise ValueError("min_correct должен быть в [0,1].")

    m = np.zeros((n, n), dtype=float)
    for j in range(n):
        good = rng.uniform(min_correct, 1.0)
        good = min(good, 1.0)
        rest = max(0.0, 1.0 - good)
        if rest > 0.0 and n > 1:
            w = rng.dirichlet(np.ones(n - 1))
            idx = [i for i in range(n) if i != j]
            m[idx, j] = rest * w
        m[j, j] = good
        s = m[:, j].sum()
        if s > 0:
            m[:, j] /= s
    return m


def forward_from_doc(
    p_x: np.ndarray,
    p_x_given_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ровно по формулам методички:

    (2.4)  p(y_j)      = Σ_i p(x_i) · p(x_i | y_j)
    (2.6)  p(x_i,y_j)  = p(y_j) · p(x_i | y_j)
    (2.7)  P(X,Y) — матрица из (2.6)

    Возвращает (p_y, p_xy). Для численной стабильности:
      • нормируем p_y и p_xy к сумме 1;
      • согласуем p_y со столбцовыми суммами p_xy.
    """
    n = p_x.shape[0]
    if p_x_given_y.shape != (n, n):
        raise ValueError("Ожидается квадратная P(X|Y) N×N и p_x длины N.")

    # (2.4) — вероятности на выходе
    p_y = p_x @ p_x_given_y
    s = p_y.sum()
    if s > 0.0:
        p_y = p_y / s

    # (2.6)–(2.7) — совместные
    p_xy = p_x_given_y * p_y[None, :]
    total = p_xy.sum()
    if total > 0.0:
        p_xy = p_xy / total
        p_y = p_xy.sum(axis=0)  # согласование p_y с матрицей

    return p_y, p_xy
