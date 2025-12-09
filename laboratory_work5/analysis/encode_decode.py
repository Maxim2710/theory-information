from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .parameters import HammingParameters
from .layout import HammingLayout


def bits_to_str(bits: np.ndarray) -> str:
    """Преобразует массив 0/1 в строку '01011...'."""
    return "".join(str(int(b)) for b in bits.tolist())


def generate_information_word(k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Генерация случайной информационной кодовой комбинации длины k
    (значения разрядов — в двоичной системе, 0 или 1).
    """
    word = rng.integers(low=0, high=2, size=k, dtype=np.int64)
    return word


def _compute_parity_indices(n_h: int, parity_pos: int) -> List[int]:
    """
    Для заданного проверочного разряда с позицией parity_pos (1,2,4,...)
    возвращает список позиций j ∈ [1..n_h], которые участвуют в соответствующей
    контрольной сумме S_i (формула (5.2)).
    """
    indices = [j for j in range(1, n_h + 1) if (j & parity_pos) != 0]
    return indices


def encode_hamming(info: np.ndarray, params: HammingParameters, layout: HammingLayout) -> np.ndarray:
    """
    Построение базового кода Хэмминга (исправление однократных ошибок).

    Шаги:
      1) Размещаем информационные разряды a_i в позициях layout.data_positions.
      2) Проверочные разряды b_0..b_{p-1} в позициях 1,2,4,.. заполняем нулями.
      3) Для каждого проверочного разряда считаем контрольную сумму S_i по (5.2) и
         выбираем b_i так, чтобы S_i = 0 (четность по модулю 2).
    """
    k = params.k
    p = params.p
    n_h = params.n_h

    if info.shape[0] != k:
        raise ValueError(f"Ожидалась информационная комбинация длины k={k}, получено {info.shape[0]}.")

    print("\n[Tx] Построение базового кода Хэмминга (исправление однократных ошибок).")
    print(f"[Tx] Информационная комбинация a (k = {k}): {bits_to_str(info)}")

    code = np.zeros(n_h, dtype=np.int64)

    # 1) Размещение информационных битов.
    print("\n[Tx] Размещение информационных разрядов по позициям (a_i → не степени двойки).")
    for idx, pos in enumerate(layout.data_positions):
        bit = int(info[idx])
        code[pos - 1] = bit
        print(f"  a_{idx + 1} = {bit} → позиция {pos}")

    # 2) Проверочные разряды пока 0.
    print("\n[Tx] Проверочные разряды b_i в позициях степеней двойки пока установлены в 0:")
    for pos in layout.parity_positions:
        print(f"  b (проверочный) в позиции {pos} = 0 (временное значение)")

    print(f"\n[Tx] Промежуточный вектор c (без окончательных b_i): {bits_to_str(code)}")

    # 3) Вычисляем b_i так, чтобы каждая контрольная сумма S_i = 0.
    print("\n[Tx] Вычисление проверочных разрядов по формулам контрольных сумм (5.2).")
    for i, pos in enumerate(sorted(layout.parity_positions)):
        parity_pos = pos
        involved = _compute_parity_indices(n_h, parity_pos)
        bits = [int(code[j - 1]) for j in involved]
        # Сначала считаем XOR по всем текущим битам (пока b_i ещё 0).
        s_val = 0
        for b in bits:
            s_val ^= b
        # Мы хотим S_i = 0, поэтому b_i устанавливаем равным текущему S_i.
        # Тогда новая сумма S_i' = S_i (старое) XOR b_i = 0.
        b_i = s_val
        code[parity_pos - 1] = b_i

        pos_list = ", ".join(str(j) for j in involved)
        print(
            f"  Контрольная сумма S_{i + 1}: позиции {{{pos_list}}}, "
            f"исходная сумма по модулю 2 = {s_val}, "
            f"выбираем b в позиции {parity_pos} = {b_i} → S_{i + 1} = 0."
        )

    print(f"\n[Tx] Итоговый базовый код Хэмминга c (длина n = {n_h}): {bits_to_str(code)}")
    return code


def add_global_parity(code_h: np.ndarray, params: HammingParameters) -> np.ndarray:
    """
    Добавление дополнительного проверочного разряда (формула (5.3)).

    Дополнительная контрольная сумма S_{p+1} соответствует глобальной чётности:
      S_{p+1} = сумма по модулю 2 всех разрядов расширенного кода.

    Построение:
      • имеем базовый код Хэмминга длины n_h;
      • добавляем один разряд b_{p} в позицию n_total так, чтобы
        сумма по модулю 2 всех разрядов расширенного кода была равна 0.
    """
    n_h = params.n_h
    n_total = params.n_total

    if code_h.shape[0] != n_h:
        raise ValueError(f"Ожидалась кодовая комбинация длины n = {n_h}, получено {code_h.shape[0]}.")

    print("\n[Tx] Модификация кода Хэмминга для обнаружения двукратных ошибок (добавление S_{p+1}).")

    # Сумма по модулю 2 всех разрядов базового кода.
    parity_h = int(code_h.sum() % 2)
    print(f"[Tx] Сумма по модулю 2 всех разрядов базового кода = {parity_h}.")

    # Добавляем дополнительный проверочный разряд так, чтобы общая чётность расширенного кода была нулевой.
    # Пусть b_global — дополнительный разряд, тогда:
    #   parity_h XOR b_global = 0 → b_global = parity_h.
    b_global = parity_h
    print(
        f"[Tx] Дополнительный проверочный разряд b_global выбирается равным {b_global}, "
        f"чтобы суммарная чётность расширенного кода была нулевой."
    )

    code_ext = np.zeros(n_total, dtype=np.int64)
    code_ext[:n_h] = code_h
    code_ext[n_h] = b_global

    print(f"[Tx] Расширенный код Хэмминга (длина n_total = {n_total}): {bits_to_str(code_ext)}")
    print(f"     Последний разряд (позиция {n_total}) — дополнительный проверочный разряд.")
    return code_ext


@dataclass
class SyndromeInfo:
    s_vector: np.ndarray  # S1..Sp
    s_int: int            # позиция ошибки по синдрому (для однократных ошибок)
    s_global: int         # S_{p+1} — глобальная чётность (0/1)


def compute_syndrome(received: np.ndarray, params: HammingParameters) -> SyndromeInfo:
    """
    Вычисление синдрома ошибки для расширенного кода:

      • S_1..S_p — контрольные суммы по формулам (5.2) для базового кода Хэмминга (по первым n_h разрядам);
      • S_{p+1}  — глобальная чётность (сумма по модулю 2 всех n_total разрядов).

    Синдром S_1..S_p интерпретируется как двоичное число с S_1 — младшим битом.
    """
    n_h = params.n_h
    n_total = params.n_total
    p = params.p

    if received.shape[0] != n_total:
        raise ValueError(f"Ожидалась принятая комбинация длины n_total = {n_total}, получено {received.shape[0]}.")

    print("\n[Rx] Вычисление синдрома ошибки (S1..Sp, S_{p+1}).")
    print(f"[Rx] Принятая комбинация r: {bits_to_str(received)}")

    s_vec = np.zeros(p, dtype=np.int64)

    # S1..Sp по первым n_h разрядам (базовый код Хэмминга).
    for i in range(p):
        parity_pos = 1 << i
        involved = _compute_parity_indices(n_h, parity_pos)
        bits = [int(received[j - 1]) for j in involved]
        s_val = 0
        for b in bits:
            s_val ^= b
        s_vec[i] = s_val
        pos_list = ", ".join(str(j) for j in involved)
        print(
            f"  S_{i + 1}: позиции {{{pos_list}}} → Σ по модулю 2 = {int(sum(bits) % 2)} → S_{i + 1} = {s_val}"
        )

    # Глобальная чётность S_{p+1}: XOR всех n_total разрядов.
    s_global = int(received.sum() % 2)
    print(f"  S_{p + 1} (глобальная чётность): сумма по модулю 2 всех разрядов = {s_global}")

    # Перевод S1..Sp в целое: S1 — младший бит, S_p — старший.
    s_int = 0
    for i, bit in enumerate(s_vec.tolist()):
        s_int |= (int(bit) & 1) << i

    print(f"[Rx] Синдром S1..Sp = {''.join(str(int(b)) for b in s_vec.tolist())} "
          f"(S1 — младший бит). Позиция ошибки по синдрому = {s_int}.")
    print(f"[Rx] Последний разряд синдрома S_(p+1) = {s_global} (кратность ошибки).")
    return SyndromeInfo(s_vector=s_vec, s_int=s_int, s_global=s_global)


@dataclass
class ErrorInfo:
    weight: int               # 0, 1 или 2 (фактическая кратность по классификации)
    kind: str                 # 'none', 'single_data', 'single_global', 'double'
    position: Optional[int]   # 1-индексированная позиция ошибки (если определена)
    description: str


def classify_error(syndrome: SyndromeInfo, params: HammingParameters) -> ErrorInfo:
    """
    Классификация ошибки по синдрому (интерпретация (5.2)–(5.3)).

    Правила:
      • Если S1..Sp = 0 и S_{p+1} = 0 → ошибок нет.
      • Если S1..Sp = 0 и S_{p+1} = 1 → ошибка только в дополнительном проверочном разряде.
      • Если S1..Sp ≠ 0 и S_{p+1} = 1 → однократная ошибка в одном из разрядов базового кода Хэмминга.
      • Если S1..Sp ≠ 0 и S_{p+1} = 0 → двукратная ошибка (обнаружена, но не исправляется).
    """
    s_vec = syndrome.s_vector
    s_int = syndrome.s_int
    s_global = syndrome.s_global
    p = params.p
    n_h = params.n_h
    n_total = params.n_total

    all_zero = np.all(s_vec == 0)

    if all_zero and s_global == 0:
        desc = "Синдром нулевой, глобальная чётность нулевая → ошибок не обнаружено."
        print(f"[Rx] {desc}")
        return ErrorInfo(weight=0, kind="none", position=None, description=desc)

    if all_zero and s_global == 1:
        # Ошибка только в дополнительном проверочном разряде (последняя позиция).
        desc = ("S1..Sp = 0, но S_(p+1) = 1 → однократная ошибка в дополнительном "
                "проверочном разряде (позиция n_total).")
        print(f"[Rx] {desc}")
        return ErrorInfo(weight=1, kind="single_global", position=n_total, description=desc)

    if not all_zero and s_global == 1:
        # Однократная ошибка в одном из первых n_h разрядов.
        if s_int == 0 or s_int > n_h:
            desc = ("S1..Sp ≠ 0 и S_(p+1) = 1, но позиция по синдрому не попадает "
                    "в диапазон 1..n → теоретически невозможная комбинация для t≤1.")
            print(f"[Rx] {desc}")
            return ErrorInfo(weight=1, kind="single_data", position=None, description=desc)
        desc = (f"S1..Sp ≠ 0 и S_(p+1) = 1 → однократная ошибка в позиции {s_int} "
                f"(1..n), определённой по синдрому.")
        print(f"[Rx] {desc}")
        return ErrorInfo(weight=1, kind="single_data", position=s_int, description=desc)

    # not all_zero and s_global == 0 → двукратная ошибка
    desc = ("S1..Sp ≠ 0 и S_(p+1) = 0 → двукратная ошибка: обнаружена, "
            "но позиция(и) ошибки не могут быть однозначно определены.")
    print(f"[Rx] {desc}")
    return ErrorInfo(weight=2, kind="double", position=None, description=desc)


@dataclass
class DecodeResult:
    received: np.ndarray
    corrected: np.ndarray
    syndrome: SyndromeInfo
    error: ErrorInfo


def correct_code(received: np.ndarray, params: HammingParameters, layout: HammingLayout) -> DecodeResult:
    """
    Декодирование и коррекция расширенного кода:

      1) Вычисляем синдром S1..Sp, S_{p+1}.
      2) Классифицируем ошибку.
      3) При однократной ошибке:
           • если она в базовом коде (позиция 1..n_h) — инвертируем этот разряд;
           • если в дополнительном проверочном разряде — инвертируем последний разряд.
         При двукратной ошибке код не корректируем.
    """
    syndrome = compute_syndrome(received, params)
    error = classify_error(syndrome, params)

    corrected = received.copy()

    if error.weight == 0:
        print("[Rx] Ошибок нет, код не изменяется.")
    elif error.weight == 1 and error.kind == "single_data" and error.position is not None:
        pos = error.position
        print(f"[Rx] Исправление однократной ошибки: инвертируем бит в позиции {pos}.")
        corrected[pos - 1] ^= 1
        print(f"[Rx] Исправленный код: {bits_to_str(corrected)}")
    elif error.weight == 1 and error.kind == "single_global" and error.position is not None:
        pos = error.position
        print(f"[Rx] Исправление однократной ошибки в дополнительном проверочном разряде: "
              f"инвертируем бит в позиции {pos}.")
        corrected[pos - 1] ^= 1
        print(f"[Rx] Исправленный код: {bits_to_str(corrected)}")
    else:
        # Двукратная ошибка или некорректируемая ситуация.
        print("[Rx] Коррекция не выполняется (двукратная или некорректируемая ошибка).")
        print(f"[Rx] Код после приёма остаётся без изменений: {bits_to_str(corrected)}")

    return DecodeResult(
        received=received,
        corrected=corrected,
        syndrome=syndrome,
        error=error,
    )


def extract_information_bits(code_ext: np.ndarray, layout: HammingLayout) -> np.ndarray:
    """
    Извлечение информационной части a из исправленной расширенной кодовой комбинации:
      • берём первые n_h разрядов (без дополнительного проверочного);
      • из них оставляем только позиции layout.data_positions.
    """
    n_total = code_ext.shape[0]
    if layout.global_parity_pos != n_total:
        raise RuntimeError("Несогласованность: позиция дополнительного разряда не совпадает с длиной кода.")

    info_bits = [int(code_ext[pos - 1]) for pos in layout.data_positions]
    return np.array(info_bits, dtype=np.int64)
