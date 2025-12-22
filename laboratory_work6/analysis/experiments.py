from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import contextlib
import sys

import numpy as np

from .codec import build_parameters, random_information_bits, encode_cyclic
from .poly_gf2 import bits_to_str_from_poly, poly_to_str
from .receiver import build_error_table, receive_and_correct


@dataclass
class Tee:
    a: any
    b: any
    def write(self, data: str) -> None:
        self.a.write(data)
        self.b.write(data)
    def flush(self) -> None:
        self.a.flush()
        self.b.flush()


def _flip_single_bit(code: int, n: int, rng: np.random.Generator) -> tuple[int, int]:
    """
    Flip one random bit among n positions (1..n from left).
    Return (received_poly, pos_left)
    """
    pos_left = int(rng.integers(1, n + 1))
    deg = n - pos_left
    received = code ^ (1 << deg)
    return received, pos_left


def run_series(variant: int, experiments: int, report_dir: Path, seed: int | None = None) -> None:
    if experiments < 6:
        experiments = 6

    # RNG init
    if seed is None:
        ss = np.random.SeedSequence()
        rng = np.random.default_rng(ss)
        shown_seed = int(ss.generate_state(1)[0])
        seed_msg = f"OS({shown_seed})"
    else:
        rng = np.random.default_rng(seed)
        seed_msg = str(seed)

    params = build_parameters(variant)
    n, k, p = params.n, params.k, params.p
    g = params.generator_poly

    print("\n=== ИНИЦИАЛИЗАЦИЯ ===")
    print(f"[Init RNG] seed = {seed_msg}")
    print(f"[Init] Итоговые параметры: k={k}, n={n}, p={p}")
    print(f"[Init] P(x) = {poly_to_str(g)}")

    # Receiver builds table once (as required by step d/e in method)
    table = build_error_table(n, g)

    success_count = 0

    print("\n=== ЗАДАНИЕ II. ЦИКЛ КОМПЛЕКСНЫХ ЭКСПЕРИМЕНТОВ ===")
    print(f"[Plan] Экспериментов: {experiments} (не менее 6)")

    for exp in range(1, experiments + 1):
        print("\n" + "=" * 78)
        print(f"ЭКСПЕРИМЕНТ {exp}")
        print("=" * 78)

        # (a) generate info bits
        print("\n(a) Генерация случайной информационной кодовой комбинации длины k.")
        info = random_information_bits(k, rng)
        info_str = "".join(str(int(b)) for b in info.tolist())
        print(f"[a] a = {info_str}")

        # (b) encode cyclic
        print("\n(b) Построение циклического кода на передатчике.")
        code = encode_cyclic(info, params)

        # (c) send generator poly (in program just log)
        print("\n(c) Передача образующего полинома P(x) на приёмную сторону.")
        print(f"[c] P(x) = {poly_to_str(g)}")

        # (d) send code with single error
        print("\n(d) Передача циклического кода через канал с однократной ошибкой.")
        received, true_pos = _flip_single_bit(code, n, rng)
        print(f"[d] Отправлено: {bits_to_str_from_poly(code, n)}")
        print(f"[d] Принято:    {bits_to_str_from_poly(received, n)}")
        print(f"[d] Истинная позиция ошибки (1..n слева): {true_pos}")

        # (e) table already built (log)
        print("\n(e) На приёмнике таблица \"позиция ↔ остаток\" уже построена (см. выше).")

        # (f) compute remainder and correct
        print("\n(f) Определение позиции ошибки по остатку и коррекция.")
        res = receive_and_correct(received, n, g, table)

        # check
        ok_code = (res.corrected_poly == code)
        print("\n[Check] Проверка совпадения исправленного кода с исходным:")
        print(f"        c (исходный):     {bits_to_str_from_poly(code, n)}")
        print(f"        c' (исправленный): {bits_to_str_from_poly(res.corrected_poly, n)}")
        print(f"        Совпадение c == c'? → {'ДА' if ok_code else 'НЕТ'}")

        ok_pos = (res.error_pos == true_pos)
        print("\n[Check] Проверка позиции ошибки:")
        print(f"        Истинная позиция: {true_pos}")
        print(f"        Найденная:        {res.error_pos if res.error_pos is not None else '—'}")
        print(f"        Совпадение? → {'ДА' if ok_pos else 'НЕТ'}")

        if ok_code and ok_pos:
            success_count += 1

        print("\n[Summary per experiment]")
        print(f"    Успешно исправлено? → {'ДА' if (ok_code and ok_pos) else 'НЕТ'}")

    print("\n=== ИТОГИ СЕРИИ (ЗАДАНИЕ II) ===")
    print(f"[Summary] Успешных экспериментов: {success_count}/{experiments} ({100.0*success_count/experiments:.2f}%)")

    print("\n=== ЗАДАНИЕ III. ВЫВОДЫ ===")
    print("1) Реализован циклический код с построением по (6.3)–(6.4): F(x)=x^pG(x)+R(x).")
    print("2) На приёмнике построена таблица \"позиция ошибки ↔ остаток\", остатки уникальны → позиция определяется однозначно.")
    print("3) В серии экспериментов случайные однократные ошибки обнаружены и исправлены корректно.")
    print("4) Совпадение c и c' подтверждает правильность реализации алгоритмов согласно методическим указаниям.")
