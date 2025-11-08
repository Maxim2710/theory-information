from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from math import log2


# ---------- Формулы из методички ----------

def entropy(p: np.ndarray) -> float:
    """
    H(Y) = - Σ p(y_i) log2 p(y_i)
    """
    p = np.asarray(p, dtype=float)
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def conditional_entropy_Y_given_Z(p_yz: np.ndarray, p_y_given_z: np.ndarray) -> float:
    """
    (3.4) H(Y|Z) = - Σ_{i,j} p(y_i, z_j) log2 p(y_i | z_j)
    """
    p_yz = np.asarray(p_yz, dtype=float)
    p_y_given_z = np.asarray(p_y_given_z, dtype=float)
    mask = p_y_given_z > 0.0
    return float(-np.sum(p_yz[mask] * np.log2(p_y_given_z[mask])))


def average_duration(p_y: np.ndarray, tau_i_microsec: np.ndarray) -> float:
    """
    τ = Σ p(y_i) * τ_i (в микросекундах)
    """
    p_y = np.asarray(p_y, dtype=float)
    tau = np.asarray(tau_i_microsec, dtype=float)
    return float(np.sum(p_y * tau))


def compute_pz_from_py_cond(p_y: np.ndarray, p_y_given_z: np.ndarray) -> np.ndarray:
    """
    (3.6) p(z_j) = Σ_i p(y_i) * p(y_i | z_j)
    """
    p_y = np.asarray(p_y, dtype=float)
    p_y_given_z = np.asarray(p_y_given_z, dtype=float)
    p_z = p_y @ p_y_given_z
    s = float(p_z.sum())
    if s > 0.0:
        p_z = p_z / s
    return p_z


def joint_from_pz_cond(p_z: np.ndarray, p_y_given_z: np.ndarray) -> np.ndarray:
    """
    (3.5) p(y_i, z_j) = p(z_j) * p(y_i | z_j)
    """
    p_z = np.asarray(p_z, dtype=float)
    p_y_given_z = np.asarray(p_y_given_z, dtype=float)
    p_yz = p_y_given_z * p_z[None, :]
    total = float(p_yz.sum())
    if total > 0.0:
        p_yz = p_yz / total
    return p_yz


# ---------- Скорости и пропускная ----------

def speed_no_noise(h_y_bits: float, tau_microsec: float) -> float:
    """
    I(Y) = H(Y) / τ   (бит/с), τ в микросекундах
    """
    return float(h_y_bits / (tau_microsec * 1e-6))


def capacity_no_noise(n: int, tau_microsec: float) -> float:
    """
    C = log2(N) / τ   (бит/с)
    """
    return float(log2(n) / (tau_microsec * 1e-6))


def speed_with_noise(h_y_bits: float, h_y_given_z_bits: float, tau_microsec: float) -> float:
    """
    I(Z,Y) = (H(Y) - H(Y|Z)) / τ
    """
    return float((h_y_bits - h_y_given_z_bits) / (tau_microsec * 1e-6))


def capacity_with_noise(n: int, h_y_given_z_bits: float, tau_microsec: float) -> float:
    """
    C = (log2(N) - H(Y|Z)) / τ
    """
    return float((log2(n) - h_y_given_z_bits) / (tau_microsec * 1e-6))


# ---------- Диагностика / инварианты ----------

@dataclass
class InvariantsReport:
    sum_py_ok: bool
    sum_pz_ok: bool
    sum_pyz_ok: bool
    cols_py_given_z_ok: bool
    yz_marginal_ok: bool
    recon_py_given_z_ok: bool


def check_invariants(p_y: np.ndarray, p_z: np.ndarray, p_yz: np.ndarray, p_y_given_z: np.ndarray) -> InvariantsReport:
    eps = 1e-12
    sum_py_ok = abs(float(p_y.sum()) - 1.0) <= 1e-9
    sum_pz_ok = abs(float(p_z.sum()) - 1.0) <= 1e-9
    sum_pyz_ok = abs(float(p_yz.sum()) - 1.0) <= 1e-9
    cols_py_given_z_ok = np.allclose(p_y_given_z.sum(axis=0), 1.0, atol=1e-12)
    yz_marginal_ok = np.allclose(p_yz.sum(axis=0), p_z, atol=1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        recon = np.divide(p_yz, p_z[None, :], out=np.zeros_like(p_yz), where=p_z[None, :] > 0)
    recon_py_given_z_ok = np.allclose(recon, p_y_given_z, atol=1e-12)

    return InvariantsReport(
        sum_py_ok=sum_py_ok,
        sum_pz_ok=sum_pz_ok,
        sum_pyz_ok=sum_pyz_ok,
        cols_py_given_z_ok=cols_py_given_z_ok,
        yz_marginal_ok=yz_marginal_ok,
        recon_py_given_z_ok=recon_py_given_z_ok,
    )
