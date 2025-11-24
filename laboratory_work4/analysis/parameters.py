from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass
class CodeParameters:
    """
    Параметры систематического кода (ЛР №4).

    k — число информационных разрядов (задано по варианту);
    p — число проверочных разрядов (исправляющих);
    n — длина кодовой комбинации (значность кода);
    d_min — минимальное кодовое расстояние (для t=1, d_min ≥ 3).
    """
    variant: int
    k: int
    p: int
    n: int
    d_min: int = 3


def variant_to_k(variant: int) -> int:
    """
    Таблица в методичке для ЛР №4 задаёт k так:

      вариант: 1  2  3 ... 54
      k:      61 60 59 ...  8

    Это задаётся простой формулой: k = 62 - variant.
    """
    if not (1 <= variant <= 54):
        raise ValueError("Номер варианта должен быть в диапазоне 1..54.")
    return 62 - variant


def choose_parity_bits(k: int, d_min: int = 3) -> tuple[int, int]:
    """
    Выбор минимального числа проверочных разрядов p и длины кода n = k + p
    для исправления всех однократных ошибок (t=1, d_min ≥ 3).

    Используем условие для кода Хэмминга с t=1:

        2^p ≥ n + 1 = k + p + 1.

    Мы подбираем минимальное p, начиная с 1, такое что 2^p ≥ k + p + 1.
    После нахождения p задаём n = k + p.

    Логи здесь — только пояснительные, на результат не влияют.
    """
    if k <= 0:
        raise ValueError("k должно быть положительным.")

    print("\n[Derive] Подбор количества проверочных разрядов p для заданного k.")
    print(f"[Derive] Требуемое минимальное расстояние d_min = {d_min} (для t=1 → d_min ≥ 3).")
    print("[Derive] Используем условие 2^p ≥ n + 1, где n = k + p.")

    p = 1
    while True:
        n = k + p
        left = 1 << p
        right = n + 1
        print(f"         Пробуем p = {p}: n = k + p = {k} + {p} = {n}, "
              f"2^p = {left}, n + 1 = {right} → "
              f"{'2^p ≥ n+1 (подходит)' if left >= right else '2^p < n+1 (недостаточно)'}")
        if left >= right:
            print(f"[Derive] Выбрано минимальное p = {p}, длина кода n = k + p = {n}.")
            break
        p += 1
    return p, k + p


def make_code_parameters(variant: int, d_min: int = 3) -> CodeParameters:
    """
    Полный расчёт параметров кода по номеру варианта.

    Возвращает CodeParameters с проверкой формулы (4.1) и d_min ≥ 3.
    """
    print("\n=== ЭТАП 1. Определение параметров кода по варианту ===")
    print(f"[Step] Номер варианта: {variant}")
    k = variant_to_k(variant)
    print(f"[Step] По таблице методички получаем число информационных разрядов k = 62 - {variant} = {k}")

    p, n = choose_parity_bits(k, d_min=d_min)

    # Проверка (4.1): 2^k (1 + n) ≤ 2^n
    lhs = (1 + n) * (1 << k)
    rhs = (1 << n)
    print("\n[Check] Дополнительная проверка условия значности (формула (4.1)):")
    print("        2^k ≤ 2^n / (1 + n)  ⇔  2^k (1 + n) ≤ 2^n")
    print(f"        2^k (1 + n) = (1 + {n}) · 2^{k} = {lhs}")
    print(f"        2^n        = 2^{n} = {rhs}")
    print(f"        Условие выполнено? → {'ДА' if lhs <= rhs else 'НЕТ'}")

    if lhs > rhs:
        raise RuntimeError(
            f"Параметры не удовлетворяют (4.1): 2^k (1+n) = {lhs} > 2^n = {rhs}"
        )

    if d_min < 3:
        raise ValueError("Для исправления всех однократных ошибок требуется d_min ≥ 3.")

    return CodeParameters(
        variant=variant,
        k=k,
        p=p,
        n=n,
        d_min=d_min,
    )


def print_parameters_info(params: CodeParameters) -> None:
    """
    Печать сводки по параметрам кода и проверке формул (4.1) и (4.2).
    """
    print("\n=== ПАРАМЕТРЫ КОДА (Задание, формулы (4.1) и (4.2)) ===")
    print(f"[Info] Вариант: {params.variant}")
    print(f"[Info] Число информационных разрядов: k = {params.k}")
    print(f"[Info] Число проверочных разрядов:    p = {params.p}")
    print(f"[Info] Значность кода:                 n = k + p = {params.k} + {params.p} = {params.n}")
    print(f"[Info] Минимальное кодовое расстояние: d_min = {params.d_min} (для t=1 → d_min ≥ 3)")

    lhs = (1 + params.n) * (1 << params.k)
    rhs = (1 << params.n)
    print("\n[Check] Формула (4.1): 2^k ≤ 2^n / (1 + n)")
    print(f"        Перепишем как 2^k (1 + n) ≤ 2^n.")
    print(f"        2^k (1 + n) = (1 + {params.n}) · 2^{params.k} = {lhs}")
    print(f"        2^n         = 2^{params.n} = {rhs}")
    print(f"        Итог: {'ВЫПОЛНЕНО' if lhs <= rhs else 'НАРУШЕНО'}")

    t = 1      # исправляем все однократные ошибки
    sigma = t  # кратность исправляемых ошибок = 1
    rhs_d = t + sigma + 1
    print("\n[Check] Формула (4.2): d_min ≥ t + σ + 1")
    print(f"        При t = σ = 1 получаем t + σ + 1 = 1 + 1 + 1 = {rhs_d}")
    print(f"        d_min = {params.d_min} → условие d_min ≥ 3: {'ВЫПОЛНЕНО' if params.d_min >= rhs_d else 'НАРУШЕНО'}")

    print("\n[Ref ] Справочная информация для синдромного декодирования:")
    print(f"       • Макс. число различных кодовых слов = 2^k = {1 << params.k}")
    print(f"       • Число возможных синдромов (включая нулевой) = 2^p = {1 << params.p}")
    print(f"       • Число позиций, которые нужно различать (включая «нет ошибки») = n + 1 = {params.n + 1}")
