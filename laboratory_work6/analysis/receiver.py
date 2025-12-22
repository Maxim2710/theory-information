# analysis/receiver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .poly_gf2 import (
    poly_to_str,
    poly_divmod_verbose,
    poly_divmod,
    bits_to_str_from_poly,
    poly_xor,
)


@dataclass
class ReceiveResult:
    received_poly: int
    syndrome: int
    error_pos: Optional[int]  # 1..n from left
    corrected_poly: int


def build_error_table(n: int, g: int) -> Dict[int, int]:
    """
    Таблица соответствия:
      позиция ошибки (1..n слева) ↔ остаток от деления x^(n-pos) на P(x).
    В таблице храним: remainder -> pos.
    """
    table: Dict[int, int] = {}
    print("\n=== ПРИЁМНИК: ПОСТРОЕНИЕ ТАБЛИЦЫ \"ПОЗИЦИЯ ОШИБКИ ↔ ОСТАТОК\" ===")
    print(f"[Rx] n={n}, P(x)={poly_to_str(g)}")
    for pos in range(1, n + 1):
        deg = n - pos
        _, r = poly_divmod(1 << deg, g)
        print(f"[Rx][Tbl] Позиция {pos} (x^{deg}) → остаток {poly_to_str(r)}")
        table[r] = pos
    return table


def receive_and_correct(
    r_poly: int,
    n: int,
    g: int,
    table: Dict[int, int],
    *,
    verbose: bool = True,
) -> ReceiveResult:
    if verbose:
        print("\n=== ПРИЁМНИК: ОПРЕДЕЛЕНИЕ ОШИБКИ И КОРРЕКЦИЯ ===")
        print(f"[Rx] Принятый код (n бит): {bits_to_str_from_poly(r_poly, n)}")
        print(f"[Rx] Принятый полином: {poly_to_str(r_poly)}")

        _, synd = poly_divmod_verbose(r_poly, g, name_a="Rcv", name_g="P")
        print(f"\n[Rx] Синдром S(x) = Rcv(x) mod P(x) = {poly_to_str(synd)}")
    else:
        _, synd = poly_divmod(r_poly, g)
        print(f"[Rx] r={bits_to_str_from_poly(r_poly, n)}")
        print(f"[Rx] S(x)={poly_to_str(synd)}")

    if synd == 0:
        if verbose:
            print("[Rx] S(x)=0 → ошибок не обнаружено.")
        else:
            print("[Rx] ok: no error")
        return ReceiveResult(received_poly=r_poly, syndrome=synd, error_pos=None, corrected_poly=r_poly)

    if synd not in table:
        if verbose:
            print("[Rx] Остаток не найден в таблице → ошибка не однократная (в рамках модели).")
        else:
            print("[Rx] fail: syndrome not in table")
        return ReceiveResult(received_poly=r_poly, syndrome=synd, error_pos=None, corrected_poly=r_poly)

    pos = table[synd]
    deg = n - pos
    e = 1 << deg
    corrected = poly_xor(r_poly, e)

    if verbose:
        print(f"[Rx] По таблице: S(x) соответствует позиции pos={pos} (x^{deg}).")
        print(f"[Rx] Исправление: инвертируем бит в позиции {pos} (XOR с x^{deg}).")
        print(f"[Rx] Исправленный полином: {poly_to_str(corrected)}")
        print(f"[Rx] Исправленный код (n бит): {bits_to_str_from_poly(corrected, n)}")
    else:
        print(f"[Rx] pos={pos}")
        print(f"[Rx] c'={bits_to_str_from_poly(corrected, n)}")

    return ReceiveResult(received_poly=r_poly, syndrome=synd, error_pos=pos, corrected_poly=corrected)
