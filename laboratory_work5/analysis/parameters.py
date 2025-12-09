from __future__ import annotations

from dataclasses import dataclass

MIN_VARIANT = 1
MAX_VARIANT = 54


@dataclass
class HammingParameters:
    """
    Параметры кода Хэмминга для ЛР №5.

    k          — число информационных разрядов (из таблицы методички).
    p          — число проверочных разрядов кода Хэмминга (исправление однократных ошибок).
    n_h        — длина кодовой комбинации базового кода Хэмминга (без дополнительного бита) n = k + p.
    n_total    — длина расширенного кода (с дополнительным проверочным разрядом) n_total = n_h + 1.
    d_min_h    — минимальное кодовое расстояние базового кода Хэмминга (исправление однократных ошибок): d_min = 3.
    d_min_ext  — минимальное кодовое расстояние модифицированного кода (обнаружение двукратных ошибок): d_min = 4.
    """
    variant: int
    k: int
    p: int
    n_h: int
    n_total: int
    d_min_h: int = 3
    d_min_ext: int = 4


def variant_to_k(variant: int) -> int:
    """
    Соответствие «номер варианта → k» для ЛР№5 (таблица в методичке):

        № варианта: 1  2  3  ...  54
        k:         8  9  10 ...  61

    Это соответствует простой формуле: k = variant + 7.
    """
    if not (MIN_VARIANT <= variant <= MAX_VARIANT):
        raise ValueError(f"Номер варианта должен быть в диапазоне {MIN_VARIANT}..{MAX_VARIANT}.")
    return variant + 7


def choose_parity_bits(k: int) -> tuple[int, int]:
    """
    Подбор числа проверочных разрядов p для кода Хэмминга с исправлением однократных ошибок.

    Условие для кода Хэмминга (t = 1):
        2^p >= n + 1,  где n = k + p.

    Подбираем минимальное p, начиная с 1.
    """
    if k <= 0:
        raise ValueError("Число информационных разрядов k должно быть > 0.")

    print("\n[Derive] Подбор числа проверочных разрядов p для кода Хэмминга.")
    print(f"[Derive] Задано k = {k} информационных разрядов.")
    print("[Derive] Используем условие 2^p ≥ n + 1, где n = k + p.")

    p = 1
    while True:
        n_h = k + p
        left = 1 << p         # 2^p
        right = n_h + 1       # n + 1
        print(
            f"  Пробуем p = {p}: n = k + p = {k} + {p} = {n_h}, "
            f"2^p = {left}, n + 1 = {right} → "
            f"{'2^p ≥ n+1 (подходит)' if left >= right else '2^p < n+1 (недостаточно)'}"
        )
        if left >= right:
            print(f"[Derive] Минимальное подходящее p = {p}, длина кода Хэмминга n = k + p = {n_h}.")
            return p, n_h
        p += 1


def make_hamming_parameters(variant: int) -> HammingParameters:
    """
    Полный расчёт параметров кода Хэмминга по номеру варианта.
    """
    print("\n=== ЭТАП 1. Определение параметров кода Хэмминга по варианту ===")
    print(f"[Step] Номер варианта: {variant}")

    k = variant_to_k(variant)
    print(f"[Step] По таблице методички для ЛР №5 получаем k = {k} информационных разрядов.")

    p, n_h = choose_parity_bits(k)
    n_total = n_h + 1

    print("\n[Step] Добавляем один дополнительный проверочный разряд для обнаружения двукратных ошибок.")
    print(f"[Step] Базовый код Хэмминга: n = k + p = {k} + {p} = {n_h}, d_min = 3 (исправление однократных ошибок).")
    print(
        f"[Step] Расширенный код (с дополнительным проверочным разрядом): "
        f"n_total = n + 1 = {n_h} + 1 = {n_total}, d_min = 4 (исправление однократных и обнаружение двукратных ошибок)."
    )

    return HammingParameters(
        variant=variant,
        k=k,
        p=p,
        n_h=n_h,
        n_total=n_total,
        d_min_h=3,
        d_min_ext=4,
    )


def print_parameters_info(params: HammingParameters) -> None:
    """
    Сводка по параметрам кода и проверка условий.
    """
    print("\n=== ПАРАМЕТРЫ КОДА ХЭММИНГА (ЛР №5) ===")
    print(f"[Info] Вариант: {params.variant}")
    print(f"[Info] Число информационных разрядов: k = {params.k}")
    print(f"[Info] Число проверочных разрядов кода Хэмминга: p = {params.p}")
    print(f"[Info] Длина базового кода Хэмминга: n = k + p = {params.k} + {params.p} = {params.n_h}")
    print(f"[Info] Длина расширенного кода (с доп. разрядом): n_total = n + 1 = {params.n_h} + 1 = {params.n_total}")
    print(f"[Info] Минимальное кодовое расстояние базового кода: d_min = {params.d_min_h}")
    print(f"[Info] Минимальное кодовое расстояние расширенного кода: d_min = {params.d_min_ext}")

    left = 1 << params.p
    right = params.n_h + 1
    print("\n[Check] Условие кода Хэмминга (исправление однократных ошибок): 2^p ≥ n + 1.")
    print(f"  2^p = 2^{params.p} = {left}")
    print(f"  n + 1 = {params.n_h} + 1 = {right}")
    print(f"  Итог: {'ВЫПОЛНЕНО' if left >= right else 'НАРУШЕНО'}")

    print("\n[Ref ] Проверочные разряды кода Хэмминга располагаются в позициях 1,2,4,8,...,2^{p-1}.")
    print("[Ref ] Информационные разряды занимают все остальные позиции от 1 до n.")
    print("[Ref ] Дополнительный проверочный разряд располагаем в последней позиции n_total и равен "
          "сумме по модулю 2 всех остальных разрядов (обеспечение d_min = 4).")
