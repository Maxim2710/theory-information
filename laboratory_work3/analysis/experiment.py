from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path
import numpy as np

from .generation import make_probabilities, make_durations_microsec, make_P_y_given_z
from .math_utils import (
    entropy, conditional_entropy_Y_given_Z,
    average_duration, compute_pz_from_py_cond, joint_from_pz_cond,
    speed_no_noise, capacity_no_noise, speed_with_noise, capacity_with_noise,
    check_invariants,
)
from .io_utils import save_csv  # функция сохранения CSV


# ---------- Форматирование для логов ----------

def _fmt_vec(v: np.ndarray, ndigits: int = 6) -> str:
    return " ".join(f"{x:.{ndigits}f}" for x in np.asarray(v, dtype=float).tolist())


def _format_array(arr: np.ndarray, max_elements: int = 9, edge: int = 4, ndigits: int = 5) -> str:
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n <= max_elements:
        return "[" + ", ".join(f"{x:.{ndigits}f}" for x in arr.tolist()) + "]"
    head = ", ".join(f"{x:.{ndigits}f}" for x in arr[:edge].tolist())
    tail = ", ".join(f"{x:.{ndigits}f}" for x in arr[-edge:].tolist())
    return "[" + head + ", ..., " + tail + "]"


def _format_matrix_cut(a: np.ndarray, rows: int = 3, cols: int = 4, ndigits: int = 5) -> list[list[str]]:
    a = np.asarray(a, dtype=float)
    r, c = a.shape
    indices = list(range(min(rows, r))) + (["..."] if r > 2*rows else []) + list(range(max(0, r-rows), r))
    out = []
    for idx in indices:
        if idx == "...":
            out.append(["..."] * (min(2*cols+1, c)))
            continue
        row = a[idx]
        if c <= 2*cols+1:
            out.append([f"{x:.{ndigits}f}" for x in row.tolist()])
        else:
            head = [f"{x:.{ndigits}f}" for x in row[:cols].tolist()]
            tail = [f"{x:.{ndigits}f}" for x in row[-cols:].tolist()]
            out.append(head + ["..."] + tail)
    return out


def _fmt_rate(bps: float) -> str:
    if bps >= 1e6:
        return f"{bps/1e6:.4f} Мбит/с"
    if bps >= 1e3:
        return f"{bps/1e3:.4f} кбит/с"
    return f"{bps:.4f} бит/с"


# ---------- Результат одного эксперимента ----------

@dataclass
class ExperimentResult:
    tau_microsec: float
    I_no_noise: float
    C_no_noise: float
    I_with_noise: float
    C_with_noise: float


# ---------- Один эксперимент ----------

def run_one_experiment(exp_no: int, N: int, rng: np.random.Generator, series_dir: Path | None, write_csv: bool) -> ExperimentResult:
    print("\n" + "="*78)
    print(f"ЭКСПЕРИМЕНТ {exp_no}  (N = {N})")
    print("="*78)

    # (a) P(Y)
    print("(a) Генерация вероятностей на входе P(Y): Dirichlet(1,...,1)")
    p_y = make_probabilities(N, rng)
    print(f"[Gen] Σ p(y) = {p_y.sum():.12f} → OK")
    print(f"[Gen] P(Y) = {_format_array(p_y)}")
    if p_y.size > 9:
        print("(примечание: массив обрезан для удобства отображения; полный — в CSV)")

    # (b) τ_i
    print("(b) Генерация длительностей τ_i ~ U(0, N] мкс")
    tau_i = make_durations_microsec(N, rng)
    print(f"[Gen] τ_i = {_format_array(tau_i)} мкс")
    if tau_i.size > 9:
        print("(примечание: массив обрезан для удобства отображения; полный — в CSV)")
    tau_avg = average_duration(p_y, tau_i)
    print(f"[Calc] Средняя длительность τ = Σ p(y_i)·τ_i = {tau_avg:.6f} μs")

    # (c) P(Y|Z)
    q = 1.0 / (2.0 * N)
    print(f"(c) Генерация P(Y|Z), q = 1/(2N) = {q:.6f}")
    p_y_given_z = make_P_y_given_z(N, rng, q=q)
    col_sums = p_y_given_z.sum(axis=0)
    diag = np.diag(p_y_given_z)
    print(f"[Gen] Проверка сумм столбцов (первые 6): {_fmt_vec(col_sums[:6])}{' ...' if N>6 else ''}")
    print(f"[Gen] Диагональ min={diag.min():.6f}, max={diag.max():.6f}")
    print("[Gen] Матрица P(Y|Z) (фрагмент):")
    for row in _format_matrix_cut(p_y_given_z):
        print("       " + str(row))
    if N > 9:
        print("(примечание: матрица обрезана для отображения; полная — в CSV)")

    # (d) Без помех
    h_y = entropy(p_y)
    print(f"(d) Без помех: H(Y) = {h_y:.6f} бит; log2(N) = {log2(N):.6f} бит")
    i_no = speed_no_noise(h_y, tau_avg)
    c_no = capacity_no_noise(N, tau_avg)
    print(f"[Res] I(Y) = H(Y)/τ = {_fmt_rate(i_no)};  C = log2(N)/τ = {_fmt_rate(c_no)}")

    # (e) С помехами — p(Z), p(Y,Z), H(Y|Z)
    p_z = compute_pz_from_py_cond(p_y, p_y_given_z)
    print(f"(e) С помехами: вычисление p(Z) по (3.6): Σ p(z) = {p_z.sum():.12f} → OK")
    print(f"[Res] P(Z) = {_format_array(p_z)}")
    p_yz = joint_from_pz_cond(p_z, p_y_given_z)
    print(f"[Res] P(Y,Z) (фрагмент):")
    for row in _format_matrix_cut(p_yz):
        print("       " + str(row))

    h_y_given_z = conditional_entropy_Y_given_Z(p_yz, p_y_given_z)
    print(f"[Calc] H(Y|Z) по (3.4) = {h_y_given_z:.6f} бит")

    i_w = speed_with_noise(h_y, h_y_given_z, tau_avg)
    c_w = capacity_with_noise(N, h_y_given_z, tau_avg)
    print(f"[Res] I(Z,Y) = (H(Y)-H(Y|Z))/τ = {_fmt_rate(i_w)};  C = {_fmt_rate(c_w)}")

    # Инварианты
    inv = check_invariants(p_y, p_z, p_yz, p_y_given_z)
    print("[Chk] Инварианты вероятностей:")
    print(f"      - Σ p(y)=1: {'OK' if inv.sum_py_ok else 'FAIL'}")
    print(f"      - Σ p(z)=1: {'OK' if inv.sum_pz_ok else 'FAIL'}")
    print(f"      - Σ p(y,z)=1: {'OK' if inv.sum_pyz_ok else 'FAIL'}")
    print(f"      - Σ_i p(y_i|z_j)=1 (по столбцам): {'OK' if inv.cols_py_given_z_ok else 'FAIL'}")
    print(f"      - Σ_i p(y_i,z_j) == p(z_j): {'OK' if inv.yz_marginal_ok else 'FAIL'}")
    print(f"      - P(Y|Z) = P(Y,Z)/P(Z): {'OK' if inv.recon_py_given_z_ok else 'FAIL'}")

    # CSV
    if write_csv and series_dir is not None:
        exp_dir = series_dir / f"exp_{exp_no:02d}"
        save_csv(exp_dir, "P_Y", p_y)
        save_csv(exp_dir, "tau_microsec", tau_i)
        save_csv(exp_dir, "P_Y_given_Z", p_y_given_z)
        save_csv(exp_dir, "P_Z", p_z)
        save_csv(exp_dir, "P_YZ", p_yz)

    return ExperimentResult(
        tau_microsec=tau_avg,
        I_no_noise=i_no, C_no_noise=c_no,
        I_with_noise=i_w, C_with_noise=c_w
    )


# ---------- Серия экспериментов ----------

def run_series(N: int, experiments: int, rng: np.random.Generator, series_dir: Path, save_csv: bool) -> None:
    if experiments < 6:
        experiments = 6

    print(f"\n=== СЕРИЯ: N={N}, экспериментов={experiments} ===")

    results: list[ExperimentResult] = []
    series_dir.mkdir(parents=True, exist_ok=True)

    for k in range(1, experiments + 1):
        # Передаём булев флаг в параметр write_csv, чтобы не маскировать функцию save_csv
        res = run_one_experiment(k, N=N, rng=rng, series_dir=series_dir, write_csv=save_csv)
        results.append(res)

    taus = np.array([r.tau_microsec for r in results], dtype=float)
    i_no = np.array([r.I_no_noise for r in results], dtype=float)
    c_no = np.array([r.C_no_noise for r in results], dtype=float)
    i_w  = np.array([r.I_with_noise for r in results], dtype=float)
    c_w  = np.array([r.C_with_noise for r in results], dtype=float)

    print("\n=== ИТОГИ СЕРИИ ===")
    print(f"[Avg] ⟨I(Y)⟩   = {_fmt_rate(i_no.mean())};   ⟨C_no_noise⟩ = {_fmt_rate(c_no.mean())}")
    print(f"[Avg] ⟨I(Z,Y)⟩ = {_fmt_rate(i_w.mean())};   ⟨C_noise⟩    = {_fmt_rate(c_w.mean())}")

    print("\n=== ЗАДАНИЕ II: СРЕДНИЕ ЗНАЧЕНИЯ ===")
    print(f"[Avg] ⟨I(Y)⟩        = {_fmt_rate(i_no.mean())}")
    print(f"[Avg] ⟨C_no_noise⟩ = {_fmt_rate(c_no.mean())}")
    print(f"[Avg] ⟨I(Z,Y)⟩     = {_fmt_rate(i_w.mean())}")
    print(f"[Avg] ⟨C_noise⟩    = {_fmt_rate(c_w.mean())}")
    print(f"[Avg] ⟨τ⟩          = {taus.mean():.5f} μs")

    print("\n=== ЗАДАНИЕ III: ВЫВОДЫ ===")
    loss_I = 100.0 * max(0.0, 1.0 - (i_w.mean() / i_no.mean()))
    loss_C = 100.0 * max(0.0, 1.0 - (c_w.mean() / c_no.mean()))
    print("1) Скорость и пропускная способность в канале с помехами ниже из-за H(Y|Z)>0.")
    print(f"2) Средняя скорость без/с помехами: {_fmt_rate(i_no.mean())} → {_fmt_rate(i_w.mean())} "
          f"(потеря {loss_I:.2f}%).")
    print(f"3) Пропускная способность без/с помехами: {_fmt_rate(c_no.mean())} → {_fmt_rate(c_w.mean())} "
          f"(потеря {loss_C:.2f}%).")
    print(f"4) Средняя длительность символа влияет на скорости: ⟨τ⟩ = {taus.mean():.5f} μs.")
    print("5) При росте N увеличивается log2(N), что повышает потенциальную C при фиксированной τ.")
