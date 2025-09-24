from __future__ import annotations

from typing import List, Tuple
from math import log2, isclose
import numpy as np

from .generation import make_probability_array
from .entropy import entropy, information, private_entropy


def _fmt_arr(values: List[float], ndigits: int = 5) -> str:
    return " ".join(f"{v:.{ndigits}f}" for v in values)


def _check_sum_to_one(probs: List[float]) -> Tuple[float, bool]:
    s = sum(probs)
    ok = isclose(s, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    return s, ok


def _self_checks(N: int) -> None:
    """Короткие санити-тесты перед серией опытов."""
    print("\n[Self] Санити-тесты...")
    uniform = [1.0 / N] * N
    h_uniform = entropy(uniform)
    print(f"[Self] Равномерное p_i=1/N: H = {h_uniform:.5f}, ожидание log2(N) = {log2(N):.5f}")

    eps = 1e-12
    delta = [1.0 - eps] + [eps / (N - 1)] * (N - 1)
    h_delta = entropy(delta)
    print(f"[Self] Почти δ-распределение: H ≈ {h_delta:.5e} (→ 0)")


def run_experiments(
    N: int,
    experiments: int = 6,
    seed: int | None = None,
    print_hmax: float | None = None,
) -> None:
    """
    Проводит не менее 6 экспериментов:
      a) генерирует вероятности p_i, i=1..N (Dirichlet)
      b) считает I(X) и H(X)
      c) печатает H_max, дефицит до максимума, частные энтропии и статистики
    seed=None -> ОС-энтропия (каждый запуск разный, сид печатается для воспроизведения)
    """
    if experiments < 6:
        experiments = 6

    # Инициализация ГСЧ: фиксированный сид или ОС-энтропия с печатью используемого значения
    if seed is None:
        ss = np.random.SeedSequence()                  # случайная энтропия из ОС
        rng = np.random.default_rng(ss)
        shown = int(ss.generate_state(1)[0])           # показываем, чем инициализировались
        print(f"[Run] Экспериментов: {experiments}; RNG seed = OS({shown})")
    else:
        rng = np.random.default_rng(seed)
        print(f"[Run] Экспериментов: {experiments}; RNG seed = {seed}")

    h_max = print_hmax if print_hmax is not None else log2(N)
    _self_checks(N)

    h_values: List[float] = []

    for k in range(1, experiments + 1):
        print(f"\n--- Эксперимент {k} ---")

        # Генерация и базовые проверки
        probs = make_probability_array(N, rng)
        s, ok = _check_sum_to_one(probs)
        print(f"[Gen] Массив вероятностей p(1..N): { _fmt_arr(probs) }")
        print(f"[Gen] Контроль суммы Σp_i = {s:.12f} -> {'OK' if ok else 'WARN'}")

        # Расчёты
        h = entropy(probs)
        i = information(probs)  # по методичке I(X) ≡ H(X)
        h_values.append(h)
        deficit = h_max - h

        # Частные энтропии и их статистики
        priv = [private_entropy(p) for p in probs]
        print(f"[Calc] Частные энтропии H(x_i): { _fmt_arr(priv) }")
        print(
            f"[Calc] min(H(x_i)) = {min(priv):.5f}, "
            f"max(H(x_i)) = {max(priv):.5f}, "
            f"avg(H(x_i)) = {sum(priv)/len(priv):.5f}"
        )

        # Инварианты
        weighted_avg_priv = sum(p * hxi for p, hxi in zip(probs, priv))
        print(f"[Chk ] 0 ≤ H(X) ≤ log2(N)? {'OK' if (0.0 <= h <= h_max + 1e-12) else 'FAIL'}")
        print(f"[Chk ] I(X) == H(X)? {'OK' if abs(i - h) <= 1e-12 else 'FAIL'}")
        print(f"[Chk ] Σ p_i·H(x_i) == H(X)? {'OK' if abs(weighted_avg_priv - h) <= 1e-12 else 'FAIL'}")

        # Вывод итогов по опыту
        print(f"[Calc] Энтропия H(X) = {h:.5f} бит")
        print(f"[Calc] Кол-во информации I(X) = {i:.5f} бит (I(X) ≡ H(X))")
        print(f"[Ref ] Максимальная H_max = {h_max:.5f} бит; дефицит H_max - H(X) = {deficit:.5f} бит")

    # Статистика по серии
    h_avg = sum(h_values) / len(h_values)
    print("\n=== Итоги серии экспериментов ===")
    print(f"[Stat] Среднее количество информации ⟨I(X)⟩ = {h_avg:.5f} бит")
    print(f"[Stat] Средняя энтропия ⟨H(X)⟩   = {h_avg:.5f} бит")
    print("[Done] Требования ЛР №1 выполнены: генерация p_i, расчёт I(X), H_max, инварианты, сводная статистика.")
