from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp

from .poly_gf2 import (
    poly_to_str, bits_to_str_from_poly, poly_mul, poly_divmod_verbose,
    poly_shift_left, poly_from_bits_str, poly_xor, poly_divmod,
)
from .parameters import (
    CyclicParameters, variant_to_k, minimal_n_by_formula_6_2, check_hamming_bound
)


def _sympy_poly_to_int(poly: sp.Poly) -> int:
    x = sp.Symbol("x")
    expr = sp.expand(poly.as_expr())
    a = 0
    for monom, coeff in sp.Poly(expr, x, modulus=2).terms():
        deg = monom[0]
        c = int(coeff) & 1
        if c:
            a |= 1 << deg
    return a


def _factor_xn_plus_1_over_gf2(n: int) -> List[Tuple[int, int, int]]:
    x = sp.Symbol("x")
    poly = sp.Poly(x**n + 1, x, modulus=2)
    coeff, factors = poly.factor_list()
    out: List[Tuple[int, int, int]] = []
    for f, e in factors:
        out.append((int(f.degree()), int(e), _sympy_poly_to_int(f)))
    return out


def _enumerate_generators_degree_p(
    factors: List[Tuple[int, int, int]],
    p: int,
) -> List[int]:
    """
    Enumerate all g(x) dividing x^n+1 with deg(g)=p by multiplying irreducible factors.
    Uses GF(2) multiplication.
    """
    out: List[int] = []

    def rec(i: int, deg_sum: int, poly_val: int) -> None:
        if deg_sum > p:
            return
        if i == len(factors):
            if deg_sum == p:
                out.append(poly_val)
            return
        deg, e, f_int = factors[i]
        # choose 0..e
        cur = poly_val
        rec(i + 1, deg_sum, cur)
        cur_pow = poly_val
        for m in range(1, e + 1):
            cur_pow = poly_mul(cur_pow, f_int)
            rec(i + 1, deg_sum + m * deg, cur_pow)

    rec(0, 0, 1)
    # remove duplicates
    uniq = list(dict.fromkeys(out))
    return uniq


def _build_syndrome_table(n: int, g: int) -> Dict[int, int]:
    """
    Map: remainder -> position(1..n from left)
    Error at position pos => error poly = x^(n-pos)
    Syndrome = x^(n-pos) mod g.
    """
    table: Dict[int, int] = {}
    for pos in range(1, n + 1):
        deg = n - pos
        _, r = poly_divmod(1 << deg, g)
        table[r] = pos
    return table


def _syndrome_table_is_unique(n: int, g: int) -> bool:
    table = _build_syndrome_table(n, g)
    # need n distinct non-zero remainders (for all positions) and none should be 0
    if len(table) != n:
        return False
    if 0 in table:
        return False
    return True


def build_parameters(variant: int) -> CyclicParameters:
    """
    Strictly according to method:
      1) k from variant,
      2) minimal n by (6.2),
      3) p = n-k,
      4) choose P(x) dividing x^n+1, degree p, and with unique syndrome table.
    If minimal n yields no valid cyclic single-error-correcting code, increase n until found.
    """
    print("\n=== ЛАБОРАТОРНАЯ РАБОТА № 6 — «ЦИКЛИЧЕСКИЙ КОД» ===")
    print("Цель: построить помехоустойчивый циклический код, обнаруживающий и исправляющий все однократные ошибки.")
    print("\n[Step] Вариант → k по таблице методички.")
    k = variant_to_k(variant)
    print(f"[Step] Вариант {variant} → k = 62 - {variant} = {k}")

    print("\n[Step] Подбор минимальной значности n по формуле (6.2):  2^k ≤ 2^n/(1+n)")
    n0 = minimal_n_by_formula_6_2(k)
    print(f"[Step] Минимальное n, удовлетворяющее (6.2), найдено: n_min = {n0}")
    print(f"       Проверка: (n+1)·2^k ≤ 2^n  →  ({n0}+1)·2^{k} ≤ 2^{n0}  →  "
          f"{'ДА' if check_hamming_bound(k, n0) else 'НЕТ'}")

    print("\n[Step] Далее требуется выбрать образующий полином P(x) степени p=n-k,")
    print("       такой что P(x) | (x^n + 1) и таблица остатков для всех позиций ошибки уникальна.")
    print("       Если для n_min это невозможно (нет подходящего делителя нужной степени / синдромы совпадают),")
    print("       увеличиваем n на 1 до первого корректного решения (по цели работы: исправление всех однократных ошибок).")

    n = n0
    while True:
        p = n - k
        print("\n" + "-" * 78)
        print(f"[Try] Пробуем n = {n} → p = n - k = {n} - {k} = {p}")

        # factor x^n + 1 over GF(2)
        print("[Try] Факторизация x^n + 1 над GF(2):")
        factors = _factor_xn_plus_1_over_gf2(n)
        for deg, e, f_int in factors:
            print(f"      (deg={deg}, кратность={e})  фактор: {poly_to_str(f_int)}")

        print(f"[Try] Перебор всех делителей степени p = {p} (произведения неприводимых факторов).")
        candidates = _enumerate_generators_degree_p(factors, p)

        print(f"[Try] Всего кандидатов g(x) степени {p}: {len(candidates)}")

        found = None
        for idx, g in enumerate(candidates, start=1):
            print("\n" + "." * 70)
            print(f"[Cand {idx}/{len(candidates)}] g(x) = {poly_to_str(g)}  (deg={p})")
            # check divisibility: (x^n + 1) mod g == 0
            xn1 = (1 << n) | 1
            _, rem = poly_divmod(xn1, g)
            print(f"[Cand] Проверка: (x^{n} + 1) mod g(x) = {poly_to_str(rem)}")
            if rem != 0:
                print("[Cand] Не делит x^n+1 → отклонён.")
                continue

            print("[Cand] Строим таблицу остатков для всех позиций ошибки и проверяем уникальность.")
            ok = _syndrome_table_is_unique(n, g)
            print(f"[Cand] Уникальные остатки для всех {n} позиций и ни один не равен 0? → {'ДА' if ok else 'НЕТ'}")
            if ok:
                found = g
                break

        if found is not None:
            print("\n" + "=" * 78)
            print("[OK] Подобраны параметры циклического кода, исправляющего все однократные ошибки.")
            print(f"     k = {k}")
            print(f"     n = {n}")
            print(f"     p = {p}")
            print(f"     P(x) = g(x) = {poly_to_str(found)}")
            print("=" * 78)
            return CyclicParameters(variant=variant, k=k, n=n, p=p, generator_poly=found)

        print(f"[Try] Для n={n} подходящий P(x) не найден. Увеличиваем n → {n+1}.")
        n += 1


def random_information_bits(k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=k, dtype=np.int64)


def info_bits_to_poly(info_bits: np.ndarray) -> int:
    s = "".join(str(int(b)) for b in info_bits.tolist())
    return poly_from_bits_str(s)


def encode_cyclic(info_bits: np.ndarray, params: CyclicParameters) -> int:
    """
    According to (6.3)-(6.4):
      G(x) from k bits,
      x^p G(x) divided by P(x),
      remainder R(x),
      F(x) = x^p G(x) + R(x).
    """
    k, n, p = params.k, params.n, params.p
    g = params.generator_poly

    print("\n=== ПЕРЕДАТЧИК: ПОСТРОЕНИЕ ЦИКЛИЧЕСКОГО КОДА ===")
    print(f"[Tx] k={k}, n={n}, p={p}")
    print(f"[Tx] P(x) = {poly_to_str(g)}")

    G = info_bits_to_poly(info_bits)
    print(f"[Tx] Информационная комбинация (битовая строка, длина k): {''.join(str(int(b)) for b in info_bits)}")
    print(f"[Tx] Соответствующий полином G(x): {poly_to_str(G)}")

    xpG = poly_shift_left(G, p)
    print(f"\n[Tx] Умножение на x^p: x^p·G(x), p={p}")
    print(f"[Tx] x^p·G(x) = {poly_to_str(xpG)}")

    Q, R = poly_divmod_verbose(xpG, g, name_a="x^p·G", name_g="P")
    print("\n[Tx] По (6.3): x^p·G(x) = P(x)·Q(x) + R(x)")
    print(f"[Tx] Q(x) = {poly_to_str(Q)}")
    print(f"[Tx] R(x) = {poly_to_str(R)}")

    F = poly_xor(xpG, R)
    print("\n[Tx] По (6.4): F(x) = x^p·G(x) + R(x)  (в GF(2) 'минус' = 'плюс')")
    print(f"[Tx] F(x) = {poly_to_str(F)}")
    print(f"[Tx] Кодовая комбинация (n бит): {bits_to_str_from_poly(F, n)}")
    print(f"     (первые k бит — информационные, последние p бит — проверочные)")

    return F
