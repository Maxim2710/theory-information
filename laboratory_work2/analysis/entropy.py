from __future__ import annotations

from typing import Sequence
import numpy as np


def _safe_log2(x: np.ndarray) -> np.ndarray:
    """
    Безопасный log2: для x == 0 возвращает 0 (так как множится на x).
    """
    out = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    out[mask] = np.log2(x[mask])
    return out


def entropy_X(p_x: Sequence[float]) -> float:
    """
    (2.1) H(X) = - Σ_i p(x_i) log2 p(x_i)
    """
    p = np.asarray(p_x, dtype=float)
    return float(-np.sum(p * _safe_log2(p)))


def conditional_entropy_X_given_Y(p_xy: np.ndarray, p_x_given_y: np.ndarray) -> float:
    """
    (2.8) H(X|Y) = - Σ_{i=1..N} Σ_{j=1..N} p(x_i, y_j) · log2 p(x_i | y_j)
    """
    if p_xy.shape != p_x_given_y.shape:
        raise ValueError("Размерности p_xy и p_x_given_y должны совпадать (N×N).")
    logs = _safe_log2(p_x_given_y)
    return float(-np.sum(p_xy * logs))


def information_incomplete_reliability(
    p_x: Sequence[float],
    p_xy: np.ndarray,
    p_x_given_y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Возвращает (H(X), H(X|Y), I(X,Y)), где (2.9) I(X,Y) = H(X) - H(X|Y).
    """
    h_x = entropy_X(p_x)
    h_x_given_y = conditional_entropy_X_given_Y(p_xy, p_x_given_y)
    return h_x, h_x_given_y, h_x - h_x_given_y
