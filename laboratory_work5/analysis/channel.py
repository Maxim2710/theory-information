from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .encode_decode import bits_to_str
from .parameters import HammingParameters


@dataclass
class ChannelResult:
    sent: np.ndarray
    received: np.ndarray
    error_weight: int          # 0, 1 или 2
    error_positions: List[int] # 1-индексированные позиции, в которых были инвертированы биты


def transmit_through_channel(code_ext: np.ndarray,
                             params: HammingParameters,
                             rng: np.random.Generator) -> ChannelResult:
    """
    Моделирование передачи кода по каналу с возможными ошибками кратности 0, 1 или 2.

    Требование ЛР:
      • кратность ошибки генерируется случайно в диапазоне [0..2];
      • позиции ошибок выбираются случайным образом по всей длине n_total, без повторений.
    """
    n_total = params.n_total
    if code_ext.shape[0] != n_total:
        raise ValueError(f"Ожидалась кодовая комбинация длины n_total = {n_total}, получено {code_ext.shape[0]}.")

    print("\n[Chan] Передача кода по каналу с возможными помехами.")
    print(f"[Chan] Передаваемый код: {bits_to_str(code_ext)}")

    # Кратность ошибки: случайное целое из {0,1,2}.
    error_weight = int(rng.integers(low=0, high=3))
    print(f"[Chan] Случайно сгенерированная кратность ошибки (0..2): {error_weight}")

    received = code_ext.copy()
    error_positions: List[int] = []

    if error_weight > 0:
        # Выбираем error_weight различных позиций из [1..n_total].
        pos_zero_based = rng.choice(n_total, size=error_weight, replace=False)
        for idx in pos_zero_based:
            pos = int(idx) + 1
            received[idx] ^= 1
            error_positions.append(pos)

        positions_str = ", ".join(str(p) for p in sorted(error_positions))
        print(f"[Chan] Инвертированы биты в позициях: {positions_str}")
    else:
        print("[Chan] В данном эксперименте ошибки отсутствуют (кратность = 0).")

    print(f"[Chan] Код на выходе канала (принятый): {bits_to_str(received)}")

    return ChannelResult(
        sent=code_ext,
        received=received,
        error_weight=error_weight,
        error_positions=sorted(error_positions),
    )
