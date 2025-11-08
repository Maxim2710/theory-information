from __future__ import annotations

import numpy as np


def make_probabilities(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    (a) Генерация P(Y): Dirichlet(1,...,1) → p_i > 0, Σ p_i = 1.
    """
    return rng.dirichlet(np.ones(n), size=1)[0]


def make_durations_microsec(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    (b) Генерация длительностей τ_i в (0, N] мкс.
    Чтобы избежать нуля, ставим маленький минимум.
    """
    eps = 1e-6
    return rng.uniform(low=eps, high=float(n), size=n)


def make_P_y_given_z(n: int, rng: np.random.Generator, q: float) -> np.ndarray:
    """
    (c) Матрица условных вероятностей P(Y|Z) размера N×N.
      • Для каждого столбца j:
        - для i != j: u_ij ~ U(0, q)
        - S = Σ_{i != j} u_ij; если S >= 1 → масштабируем к (1 - ε)
        - диагональ: 1 - Σ offdiag
      • По столбцам сумма = 1.
      • Все offdiag ≤ q, diag ≥ 0.
    """
    eps = 1e-12
    m = np.zeros((n, n), dtype=float)
    for j in range(n):
        off = rng.uniform(low=0.0, high=q, size=n-1)
        s = float(off.sum())
        if s >= 1.0 - eps:
            off *= (1.0 - 1e-6) / s
            s = float(off.sum())
        diag = 1.0 - s
        m[:, j] = np.insert(off, j, diag)
    # Нормировка (на всякий случай численно)
    col_sums = m.sum(axis=0)
    m = np.divide(m, col_sums[None, :], out=np.zeros_like(m), where=col_sums[None, :] > 0)
    return m
