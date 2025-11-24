from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from .parameters import CodeParameters


@dataclass
class CodeMatrices:
    """
    Набор матриц для систематического (n, k) кода:

      G  — производящая матрица P_{n,k} размера k×n (в виде [I_k | H_p]);
      H  — проверочная матрица H размера p×n ([H1 | I_p]);
      H1 — подматрица H1 размера p×k (транспонирована к H_p);
      Hp — проверочная подматрица H_p размера k×p.

    Все матрицы над GF(2), храним как массивы int64 с элементами 0 или 1.
    """
    G: np.ndarray
    H: np.ndarray
    H1: np.ndarray
    Hp: np.ndarray


def _int_to_bits_vector(x: int, p: int) -> np.ndarray:
    """
    Перевод целого x (1..2^p-1) в столбец из p бит (numpy).

    Порядок бит: i-й бит (0 — младший) идёт в i-ю строку вектора.
    Один и тот же порядок используется и при построении H, и при вычислении синдрома,
    поэтому соответствие «столбец H ↔ синдром ↔ позиция ошибки» сохраняется.
    """
    bits = [(x >> i) & 1 for i in range(p)]
    return np.array(bits, dtype=np.int64)  # размер p×1 (как столбец)


def build_matrices(params: CodeParameters) -> CodeMatrices:
    """
    Построение производящей матрицы G=P_{n,k} и проверочной матрицы H по методике:

      1) Выбираем p так, чтобы 2^p ≥ n + 1 (выполнено в CodeParameters).
      2) Строим все ненулевые p-разрядные векторы (1..2^p - 1).
      3) Выделяем p столбцов под проверочные разряды: векторы e_1,..,e_p (столбцы единичной матрицы I_p).
      4) Остальные ненулевые векторы используем как столбцы H1 (p×k) для информационных разрядов,
         выбираем первые k из них (кроме e_i, чтобы столбцы были различны и ненулевы).
      5) Получаем H = [H1 | I_p] (формула (4.8)), размер p×n.
      6) Получаем H_p как транспонированную к H1: H_p = H1^T (формулы (4.4)–(4.5)).
      7) Строим производящую матрицу систематического кода:
           G = [I_k | H_p]  (формула (4.4)).

    Логика работы — как раньше, добавлены только подробные пояснения в лог.
    """
    k = params.k
    p = params.p
    n = params.n

    print("\n=== ЭТАП 2. Построение производящей и проверочной матриц (Задание I) ===")
    print(f"[Build] k = {k}, p = {p}, n = {n}")
    print("[Build] H будет иметь вид [H1 | I_p], G будет иметь вид [I_k | H_p], "
          "где H_p = H1^T (формулы (4.4)–(4.8)).")

    # 1) Все ненулевые p-разрядные векторы (целые 1..2^p-1)
    all_nonzero = list(range(1, 1 << p))
    print(f"\n[Build] Общее количество ненулевых p-разрядных столбцов: 2^p - 1 = {len(all_nonzero)}")

    # 2) p стандартных столбцов (единичная матрица I_p)
    parity_cols_int: List[int] = [1 << i for i in range(p)]  # 1,2,4,...,2^{p-1}
    print(f"[Build] Столбцы для единичной подматрицы I_p (проверочные разряды) в целочисленном виде:")
    print(f"        {parity_cols_int}")

    for val in parity_cols_int:
        assert val in all_nonzero

    # 3) Столбцы для информационной части H1: любые ненулевые, кроме parity_cols_int
    info_candidates = [v for v in all_nonzero if v not in parity_cols_int]
    if len(info_candidates) < k:
        raise RuntimeError("Недостаточно ненулевых столбцов для построения H1 (увеличьте p).")

    info_cols_int = info_candidates[:k]
    print("\n[Build] Столбцы для информационной части H1 (в виде целых чисел):")
    print(f"        (показаны первые min(k,10)) → {info_cols_int[:10]}")
    print("        Каждый такой столбец задаёт набор единиц в определённых строках проверочной матрицы H.")

    # Строим H1 (p×k): каждый столбец — p-битный вектор
    H1 = np.zeros((p, k), dtype=np.int64)
    for j, val in enumerate(info_cols_int):
        H1[:, j] = _int_to_bits_vector(val, p)

    print("\n[Build] Подматрица H1 (p×k), задающая, какие информационные разряды входят в проверочные суммы:")
    rows_to_show = min(p, 6)
    cols_to_show = min(k, 10)
    for i in range(rows_to_show):
        row = " ".join(str(int(x)) for x in H1[i, :cols_to_show])
        if cols_to_show < k:
            row += " ..."
        print(f"   H1[строка {i+1}] = {row}")

    # Строим I_p (правая часть H)
    I_p = np.eye(p, dtype=np.int64)
    print("\n[Build] Подматрица I_p (p×p) — единичная, задаёт проверочные разряды как отдельные символы:")

    for i in range(min(p, 6)):
        row = " ".join(str(int(x)) for x in I_p[i])
        print(f"   I_p[строка {i+1}] = {row}")

    # Проверочная матрица H = [H1 | I_p], размер p×n
    H = np.hstack([H1, I_p])
    print(f"\n[Build] Проверочная матрица H (p×n) построена как [H1 | I_p], размер {H.shape[0]}×{H.shape[1]}.")

    # Подматрица H_p (k×p) — транспонированная к H1
    Hp = H1.T.copy()
    print(f"[Build] Проверочная подматрица H_p = H1^T (размер {Hp.shape[0]}×{Hp.shape[1]}).")

    # Производящая матрица G = [I_k | H_p], размер k×n
    I_k = np.eye(k, dtype=np.int64)
    G = np.hstack([I_k, Hp])
    print(f"[Build] Производящая матрица G = [I_k | H_p], размер {G.shape[0]}×{G.shape[1]}.")

    # Базовые проверки: линейность, систематичность и однозначность синдромов
    _validate_matrices(params, G, H)

    print("\n[Map ] Отображение «позиция ошибки ↔ столбец H ↔ синдром (в двоичном и целочисленном виде)» "
          "для первых нескольких разрядов:")
    p_bits = p
    preview_cols = min(n, 10)
    for j in range(preview_cols):
        col = H[:, j]
        # Перевод в int по тому же правилу, что и синдром
        val = 0
        for i, bit in enumerate(col.tolist()):
            val |= (int(bit) & 1) << i
        print(f"   Позиция {j+1}: столбец H[:,{j}] = {''.join(str(int(b)) for b in col.tolist())} "
              f"→ синдром = {val} (десятичное)")

    return CodeMatrices(G=G, H=H, H1=H1, Hp=Hp)


def _validate_matrices(params: CodeParameters, G: np.ndarray, H: np.ndarray) -> None:
    """
    Минимальный набор проверок:
      - размерности G и H соответствуют k×n и p×n;
      - производящая и проверочная матрицы согласованы: H·G^T = 0 (все кодовые слова удовлетворяют проверочным уравнениям);
      - все столбцы H ненулевые и попарно различны → d_min ≥ 3 (однократные ошибки однозначно определяются синдромом).
    """
    k, p, n = params.k, params.p, params.n

    print("\n[Check] Внутренняя проверка согласованности матриц G и H:")

    if G.shape != (k, n):
        raise RuntimeError(f"Ожидается G размера {k}×{n}, получено {G.shape}.")
    if H.shape != (p, n):
        raise RuntimeError(f"Ожидается H размера {p}×{n}, получено {H.shape}.")
    print("        - Размерности G и H соответствуют ожидаемым: ОК")

    # Проверка H · G^T = 0 над GF(2)
    left = (H @ G.T) % 2
    if not np.array_equal(left, np.zeros_like(left)):
        raise RuntimeError("Нарушено соотношение H·G^T = 0 (матрицы не согласованы).")
    print("        - Соотношение H·G^T ≡ 0 (над GF(2)) выполняется: ОК")

    # Проверка ненулевости и различия столбцов H (условия для d_min ≥ 3 при t=1)
    cols = [tuple(H[:, j].tolist()) for j in range(n)]
    if any(all(bit == 0 for bit in col) for col in cols):
        raise RuntimeError("Обнаружен нулевой столбец в H — это запрещено.")
    if len(set(cols)) != n:
        raise RuntimeError("Обнаружены совпадающие столбцы в H — синдромы перестанут быть однозначными.")

    print("        - Все столбцы H ненулевые и попарно различны → d_min ≥ 3 для t=1: ОК")


def preview_matrices(m: CodeMatrices, max_rows: int = 6, max_cols: int = 10) -> None:
    """
    Краткий текстовый просмотр матриц G и H (обрезанный) для логов «Ход работы, Задание I».
    """
    def _preview(name: str, mat: np.ndarray) -> None:
        r, c = mat.shape
        print(f"\n[{name}] размер {r}×{c} (показаны не все элементы для удобства)")
        rows_to_show = min(max_rows, r)
        cols_to_show = min(max_cols, c)
        for i in range(rows_to_show):
            row = " ".join(str(int(x)) for x in mat[i, :cols_to_show])
            if cols_to_show < c:
                row += " ..."
            print(f"   строка {i+1}: {row}")
        if rows_to_show < r:
            print("   ... (остальные строки опущены)")

    _preview("G = P_{n,k} (производящая матрица)", m.G)
    _preview("H (проверочная матрица)", m.H)
