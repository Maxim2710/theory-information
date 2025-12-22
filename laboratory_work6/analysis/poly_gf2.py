from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# =========================
# GF(2) polynomial utilities
# Representation: integer bits.
# Bit i = coefficient of x^i (LSB -> x^0).
# =========================

def poly_degree(a: int) -> int:
    return a.bit_length() - 1


def poly_is_zero(a: int) -> bool:
    return a == 0


def poly_to_terms(a: int) -> List[int]:
    out: List[int] = []
    i = 0
    while a:
        if a & 1:
            out.append(i)
        a >>= 1
        i += 1
    return out


def poly_to_str(a: int) -> str:
    if a == 0:
        return "0"
    terms = poly_to_terms(a)
    parts = []
    for t in reversed(terms):
        if t == 0:
            parts.append("1")
        elif t == 1:
            parts.append("x")
        else:
            parts.append(f"x^{t}")
    return " + ".join(parts)


def bits_to_str_from_poly(a: int, n: int) -> str:
    # print MSB..LSB of length n (degree n-1..0)
    return "".join("1" if (a >> i) & 1 else "0" for i in range(n - 1, -1, -1))


def poly_from_bits_str(bitstr: str) -> int:
    # bitstr MSB..LSB (left->highest degree)
    a = 0
    n = len(bitstr)
    for idx, ch in enumerate(bitstr):
        if ch == "1":
            deg = n - 1 - idx
            a |= 1 << deg
    return a


def poly_xor(a: int, b: int) -> int:
    return a ^ b


def poly_mul(a: int, b: int) -> int:
    res = 0
    i = 0
    bb = b
    while bb:
        if bb & 1:
            res ^= a << i
        bb >>= 1
        i += 1
    return res


def poly_divmod(a: int, g: int) -> Tuple[int, int]:
    if g == 0:
        raise ZeroDivisionError("division by zero polynomial")

    q = 0
    r = a
    dg = poly_degree(g)
    while r != 0 and poly_degree(r) >= dg:
        shift = poly_degree(r) - dg
        q ^= 1 << shift
        r ^= g << shift
    return q, r


def poly_divmod_verbose(a: int, g: int, *, name_a: str, name_g: str) -> Tuple[int, int]:
    """
    Long division in GF(2) with FULL logging.
    """
    print("\n[GF2 DIV] Деление многочленов (по модулю 2) \"в столбик\":")
    print(f"          Делимое {name_a}(x) = {poly_to_str(a)}")
    print(f"          Делитель {name_g}(x) = {poly_to_str(g)}")
    print(f"          deg({name_a}) = {poly_degree(a)}; deg({name_g}) = {poly_degree(g)}")

    if g == 0:
        raise ZeroDivisionError("division by zero polynomial")

    q = 0
    r = a
    dg = poly_degree(g)
    step = 0
    while r != 0 and poly_degree(r) >= dg:
        step += 1
        dr = poly_degree(r)
        shift = dr - dg
        term = 1 << shift
        print(f"\n          Шаг {step}:")
        print(f"            Текущий остаток R_{step-1}(x) = {poly_to_str(r)}")
        print(f"            deg(R_{step-1}) = {dr} >= {dg} → продолжаем.")
        print(f"            Старший член частного: x^{shift}")
        q ^= term
        sub = g << shift
        print(f"            Вычитаем (по mod2 это XOR): {name_g}(x)·x^{shift} = {poly_to_str(sub)}")
        r ^= sub
        print(f"            Новый остаток R_{step}(x) = R_{step-1}(x) + {name_g}(x)·x^{shift} = {poly_to_str(r)}")
        print(f"            Частное на данный момент Q(x) = {poly_to_str(q)}")

    print("\n[GF2 DIV] Итог деления:")
    print(f"          Частное Q(x) = {poly_to_str(q)}")
    print(f"          Остаток R(x) = {poly_to_str(r)}")
    print(f"          Проверка: {name_a}(x) = {name_g}(x)·Q(x) + R(x) (в GF(2))")
    return q, r


def poly_shift_left(a: int, p: int) -> int:
    return a << p
