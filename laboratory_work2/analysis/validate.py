from __future__ import annotations

import numpy as np


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def _conditional_entropy_from_pxy(p_xy: np.ndarray, axis: int) -> float:
    """
    axis=1 → H(X|Y)  = -Σ p(x,y) log p(x|y)
    axis=0 → H(Y|X)  = -Σ p(x,y) log p(y|x)
    """
    p_xy = np.asarray(p_xy, dtype=float)
    if axis == 1:
        p_y = p_xy.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_x_given_y = np.divide(p_xy, p_y[None, :], out=np.zeros_like(p_xy), where=p_y[None, :] > 0)
        mask = p_x_given_y > 0
        return float(-np.sum(p_xy[mask] * np.log2(p_x_given_y[mask])))
    else:
        p_x = p_xy.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_y_given_x = np.divide(p_xy, p_x[:, None], out=np.zeros_like(p_xy), where=p_x[:, None] > 0)
        mask = p_y_given_x > 0
        return float(-np.sum(p_xy[mask] * np.log2(p_y_given_x[mask])))


def _mutual_info_from_pxy(p_xy: np.ndarray) -> float:
    """
    I(X;Y) = Σ_{i,j} p(x_i,y_j) · log2( p(x_i,y_j) / (p(x_i)p(y_j)) )
    """
    p_xy = np.asarray(p_xy, dtype=float)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(p_x, p_y)
        ratio = np.divide(p_xy, denom, out=np.ones_like(p_xy), where=denom > 0)
        mask = (p_xy > 0) & (denom > 0)
        return float(np.sum(p_xy[mask] * np.log2(ratio[mask])))


def quick_checks(
    p_x: np.ndarray,
    p_x_given_y: np.ndarray,
    p_y: np.ndarray,
    p_xy: np.ndarray,
    min_correct: float = 0.70,
) -> None:
    """
    Жёсткие инварианты и требования:
      • p_x, p_y, p_xy — нормированы и неотрицательны;
      • столбцы P(X|Y) суммируются к 1;
      • диагональ P(X|Y) ≥ min_correct;
      • P(Y) совпадает с Σ_i P(X,Y)[i,j];
      • P(X|Y) восстанавливается как P(X,Y)/P(Y).
    """
    eps = 1e-10
    n = len(p_x)
    assert p_x.shape == (n,)
    assert p_y.shape == (n,)
    assert p_x_given_y.shape == (n, n)
    assert p_xy.shape == (n, n)

    assert np.all(p_x >= -eps) and np.all(p_y >= -eps) and np.all(p_xy >= -eps)
    assert np.isclose(p_x.sum(), 1.0, atol=eps)
    assert np.isclose(p_y.sum(), 1.0, atol=eps)
    assert np.isclose(p_xy.sum(), 1.0, atol=eps)
    assert np.allclose(p_x_given_y.sum(axis=0), 1.0, atol=eps)

    diag = np.diag(p_x_given_y)
    assert float(diag.min()) >= (min_correct - 1e-12)

    assert np.allclose(p_xy.sum(axis=0), p_y, atol=1e-12)

    recon = np.divide(p_xy, p_y[None, :], out=np.zeros_like(p_xy), where=p_y[None, :] > 0)
    assert np.allclose(recon, p_x_given_y, atol=1e-12)


def diagnostics_block(p_x: np.ndarray, p_y: np.ndarray, p_xy: np.ndarray) -> None:
    """
    Диагностика для отчёта:
      • H(X) по входу и по маргинале из P(X,Y);
      • H(X|Y), H(Y|X);
      • I(X;Y) по формуле Σ p log ratio.
    """
    hx_in = _entropy(p_x)
    px_from_xy = p_xy.sum(axis=1)
    hx_from_xy = _entropy(px_from_xy)
    hxy = _conditional_entropy_from_pxy(p_xy, axis=1)
    hyx = _conditional_entropy_from_pxy(p_xy, axis=0)
    imi = _mutual_info_from_pxy(p_xy)

    print(f"[Диаг] H(X) по входу      = {hx_in:.6f} бит")
    print(f"[Диаг] H(X) из P(X,Y)     = {hx_from_xy:.6f} бит; |Δ|={abs(hx_in - hx_from_xy):.6e}")
    print(f"[Диаг] H(X|Y)={hxy:.6f} бит; H(Y|X)={hyx:.6f} бит")
    print(f"[Диаг] I(X;Y) (через Σ p log ratio) = {imi:.6f} бит")
