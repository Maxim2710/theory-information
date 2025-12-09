from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .parameters import HammingParameters


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dataclass
class HammingLayout:
    """
    Описание расположения разрядов в коде Хэмминга.

    data_positions       — позиции информационных разрядов a_i (1..n_h).
    parity_positions     — позиции проверочных разрядов b_0..b_{p-1} (1,2,4,...).
    global_parity_pos    — позиция дополнительного проверочного разряда (последний разряд расширенного кода).
    """
    data_positions: List[int]
    parity_positions: List[int]
    global_parity_pos: int


def build_layout(params: HammingParameters) -> HammingLayout:
    """
    Формируем распределение разрядов для кода Хэмминга:

    • Позиции 1,2,4,8,...,2^{p-1} — проверочные разряды b_0..b_{p-1}.
    • Остальные позиции 1..n_h — информационные разряды a_1..a_k.
    • Позиция n_total = n_h + 1 — дополнительный проверочный разряд (глобальная чётность).
    """
    data_positions: List[int] = []
    parity_positions: List[int] = []

    for pos in range(1, params.n_h + 1):
        if _is_power_of_two(pos):
            parity_positions.append(pos)
        else:
            data_positions.append(pos)

    if len(data_positions) != params.k:
        raise RuntimeError(
            f"Несогласованность: число информационных позиций = {len(data_positions)}, "
            f"ожидалось k = {params.k}."
        )
    if len(parity_positions) != params.p:
        raise RuntimeError(
            f"Несогласованность: число проверочных позиций = {len(parity_positions)}, "
            f"ожидалось p = {params.p}."
        )

    layout = HammingLayout(
        data_positions=data_positions,
        parity_positions=parity_positions,
        global_parity_pos=params.n_total,
    )
    return layout


def print_layout(layout: HammingLayout) -> None:
    """
    Печать распределения разрядов (формула (5.1) из методички в общем виде).
    """
    print("\n=== РАСПРЕДЕЛЕНИЕ РАЗРЯДОВ В КОДЕ ХЭММИНГА (обобщённый вариант (5.1)) ===")
    print("[Layout] Позиции проверочных разрядов (b_i): "
          + ", ".join(str(p) for p in layout.parity_positions))
    print("[Layout] Позиции информационных разрядов (a_i): "
          + ", ".join(str(p) for p in layout.data_positions))
    print(f"[Layout] Позиция дополнительного проверочного разряда (глобальная чётность): "
          f"{layout.global_parity_pos}")

    max_pos = layout.global_parity_pos
    print("\n[Layout] Схема позиций (1..n_total), где 'b' — проверочный, 'a' — информационный, 'B' — доп. проверочный:")
    marks = []
    for pos in range(1, max_pos + 1):
        if pos == layout.global_parity_pos:
            marks.append("B")
        elif pos in layout.parity_positions:
            marks.append("b")
        else:
            marks.append("a")
    positions_line = " ".join(f"{pos:2d}" for pos in range(1, max_pos + 1))
    roles_line = " ".join(f" {m}" for m in marks)
    print(" Позиции: " + positions_line)
    print(" Роли   : " + roles_line)
