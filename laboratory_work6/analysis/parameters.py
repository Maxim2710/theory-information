from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import sympy as sp


@dataclass
class CyclicParameters:
    variant: int
    k: int
    n: int
    p: int
    generator_poly: int  # GF(2) polynomial as int


def variant_to_k(variant: int) -> int:
    # table: k = 62 - variant
    if not (1 <= variant <= 54):
        raise ValueError("Вариант должен быть 1..54")
    return 62 - variant


def check_hamming_bound(k: int, n: int) -> bool:
    # (6.2) in method form: 2^k <= 2^n/(n+1)  <=> (n+1)*2^k <= 2^n
    return (n + 1) * (1 << k) <= (1 << n)


def minimal_n_by_formula_6_2(k: int) -> int:
    n = k + 1
    while not check_hamming_bound(k, n):
        n += 1
    return n


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
    """
    Return list of irreducible factors of x^n + 1 over GF(2) as:
      (degree, multiplicity, factor_int)
    """
    x = sp.Symbol("x")
    poly = sp.Poly(x**n + 1, x, modulus=2)
    coeff, factors = poly.factor_list()
    out: List[Tuple[int, int, int]] = []
    for f, e in factors:
        out.append((int(f.degree()), int(e), _sympy_poly_to_int(f)))
    return out


def _all_degree_p_divisors_from_factors(factors: List[Tuple[int, int, int]], target_deg: int) -> List[int]:
    """
    Enumerate all products (with multiplicities) whose total degree == target_deg.
    Each factor can be taken from 0..e times.
    """
    out: List[int] = []

    def rec(i: int, current_deg: int, current_poly: int) -> None:
        if current_deg > target_deg:
            return
        if i == len(factors):
            if current_deg == target_deg:
                out.append(current_poly)
            return

        deg, e, f_int = factors[i]
        # take 0..e
        poly_pow = 1  # polynomial "1"
        for m in range(e + 1):
            rec(i + 1, current_deg + m * deg, current_poly if m == 0 else current_poly * 0 + 0)  # overwritten below
            # NOTE: We cannot use '*' (integer) for polynomials; we build below in-place in main recursion
            break

    # Implement properly with GF2 multiplication in caller to avoid circular import.
    # Here just placeholder; actual enumeration done in build_parameters() in codec module.
    return out


def print_parameter_stage_header() -> None:
    print("\n=== ЗАДАНИЕ I. ОПРЕДЕЛЕНИЕ n, p И ВЫБОР P(x) ===")
