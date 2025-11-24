from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .matrices import CodeMatrices


def bits_to_str(bits: np.ndarray) -> str:
    """
    Преобразует одномерный массив 0/1 в строку вида '01011...'.
    """
    return "".join(str(int(b)) for b in bits.tolist())


def random_information_word(k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Генерация случайной информационной кодовой комбинации длины k
    (значения разрядов — в двоичной системе, 0 или 1).
    """
    word = rng.integers(low=0, high=2, size=k, dtype=np.int64)
    return word


def encode(info: np.ndarray, matrices: CodeMatrices) -> np.ndarray:
    """
    Кодирование (формирование систематического кода) по производящей матрице G:

      c = a · G (по модулю 2),

    где a — строковый вектор информационных битов длины k,
        G — матрица k×n,
        c — строковый вектор длины n (систематический код).

    Логи подробно показывают разбиение на информационную и проверочную части.
    """
    if info.ndim != 1:
        raise ValueError("Информационный вектор должен быть одномерным.")
    if info.shape[0] != matrices.G.shape[0]:
        raise ValueError("Длина информационного вектора не совпадает с k.")

    k = matrices.G.shape[0]
    p = matrices.H.shape[0]
    n = matrices.G.shape[1]

    print("\n[Enc] Кодирование сообщения на передатчике:")
    print(f"      Длина информационной части k = {k}, длина кода n = {n}, проверочных разрядов p = {p}.")
    print(f"      Информационный вектор a: {bits_to_str(info)}")

    # Превращаем info в строковый вектор 1×k
    row = info.reshape(1, -1)
    code = (row @ matrices.G) % 2
    code = code.reshape(-1)

    data_part = code[:k]
    parity_part = code[k:]

    print("[Enc] Производящая матрица G умножена на a (по модулю 2): c = a·G.")
    print(f"      Кодовая комбинация c:    {bits_to_str(code)}")
    print(f"      Информационная часть:    {bits_to_str(data_part)} (первые k разрядов)")
    print(f"      Проверочная часть (p б): {bits_to_str(parity_part)} (последние p разрядов)")

    return code


@dataclass
class ChannelResult:
    """
    Результат прохождения через канал с однократной ошибкой.
    """
    sent_code: np.ndarray         # исходная кодовая комбинация (n бит)
    received_code: np.ndarray     # комбинация на приёмнике (искажённая)
    error_position: int           # позиция ошибки (0..n-1)


def transmit_with_single_error(code: np.ndarray, rng: np.random.Generator) -> ChannelResult:
    """
    Передача систематического кода через канал с однократной ошибкой:
      - случайным образом выбирается позиция ошибки (0..n-1);
      - соответствующий бит инвертируется.

    Возвращает исходный код, принятый код и индекс позиции ошибки.
    """
    n = code.shape[0]
    pos = int(rng.integers(low=0, high=n))

    print("\n[Chan] Передача по каналу с однократной ошибкой:")
    print(f"       Длина кодовой комбинации n = {n}.")
    print(f"       Случайно выбрана позиция ошибки (1..n): {pos + 1}")
    print(f"       Исходный код перед передачей:  {bits_to_str(code)}")

    received = code.copy()
    received[pos] ^= 1

    print(f"       Принятый код после искажения: {bits_to_str(received)}")
    print("       (один бит инвертирован в выбранной позиции)")

    return ChannelResult(sent_code=code.copy(), received_code=received, error_position=pos)


def compute_syndrome(received: np.ndarray, matrices: CodeMatrices) -> np.ndarray:
    """
    Вычисление синдрома ошибки:

        s = r · H^T  (по модулю 2),

    где r — принятая кодовая комбинация (1×n),
        H — проверочная матрица (p×n),
        s — синдром длины p. Нулевой синдром соответствует отсутствию ошибок.

    Здесь дополнительно выводятся подробные составляющие каждой проверочной суммы.
    """
    if received.ndim != 1:
        raise ValueError("Принятый вектор должен быть одномерным.")
    H = matrices.H
    p, n = H.shape

    print("\n[Dec] Вычисление синдрома ошибки на приёмной стороне (s = r·H^T):")
    print(f"      Принятый вектор r: {bits_to_str(received)}")
    print(f"      Размер проверочной матрицы H: {p}×{n} (p строк, n столбцов).")
    print("      Каждая строка H задаёт одно контрольное равенство по модулю 2.")

    # Пошагово считаем каждую компоненту синдрома как сумму по модулю 2
    syndrome = np.zeros(p, dtype=np.int64)
    for row_idx in range(p):
        indices = np.where(H[row_idx, :] == 1)[0]
        involved_bits = received[indices]
        s_val = int(involved_bits.sum() % 2)
        syndrome[row_idx] = s_val
        pos_list = ", ".join(str(i + 1) for i in indices.tolist())
        print(f"      s_{row_idx + 1} = Σ r_i (по позициям i ∈ {{{pos_list}}}) mod 2 "
              f"= {int(involved_bits.sum())} mod 2 = {s_val}")

    print(f"      Итоговый синдром s: {bits_to_str(syndrome)}")

    return syndrome


def find_error_position_by_syndrome(syndrome: np.ndarray, matrices: CodeMatrices) -> Optional[int]:
    """
    По синдрому (p бит) находим позицию ошибки, сравнивая его со столбцами H.

    Каждый столбец j матрицы H равен синдрому ошибки, возникшей в j-й позиции.
    Если syndrome == 0 → ошибок нет.
    Если syndrome совпал с каким-то столбцом H[:, j] → ошибка в позиции j.
    Если совпадения нет → ошибка некорректируемая (кратность > 1 или вне модели).
    """
    H = matrices.H
    p, n = H.shape

    if np.all(syndrome == 0):
        print("[Dec] Синдром равен нулю → ошибок не обнаружено (идеальная передача).")
        return None

    print("[Dec] Поиск позиции одиночной ошибки по синдрому.")
    print(f"      Синдром s: {''.join(str(int(b)) for b in syndrome.tolist())}")

    target_val = 0
    for i, bit in enumerate(syndrome.tolist()):
        target_val |= (int(bit) & 1) << i
    print(f"      Синдром в десятичном виде: {target_val}")

    for j in range(n):
        col = H[:, j]
        col_val = 0
        for i, bit in enumerate(col.tolist()):
            col_val |= (int(bit) & 1) << i
        if np.array_equal(col, syndrome):
            print(f"      Найден столбец H[:, {j}] = s (десятичное значение {col_val}).")
            print(f"      Это означает: ошибка в разряде j = {j} (позиция {j + 1} от начала кода).")
            return j

    print("      Ни один столбец H не совпал с синдромом — ошибка не является однократной "
          "в рамках данной модели (некорректируемая).")
    return None


@dataclass
class DecodeResult:
    """
    Результат декодирования на приёмнике.
    """
    received: np.ndarray          # принятая комбинация
    corrected: np.ndarray         # исправленная комбинация
    syndrome: np.ndarray          # синдром ошибки
    syndrome_int: int             # синдром как целое число (для справки)
    detected_error_pos: Optional[int]  # позиция ошибки по синдрому (0..n-1) или None
    had_error: bool               # признак: был ли ненулевой синдром


def decode(received: np.ndarray, matrices: CodeMatrices) -> DecodeResult:
    """
    Декодирование:
      1) вычисляем синдром s = r·H^T;
      2) если s == 0 → ошибок нет;
      3) иначе ищем позицию j, для которой H[:, j] == s и инвертируем бит j.

    Возвращает структуру DecodeResult.
    """
    s = compute_syndrome(received, matrices)

    # Представление синдрома как числа (LSB — младший бит)
    syndrome_int = 0
    for i, bit in enumerate(s.tolist()):
        syndrome_int |= (int(bit) & 1) << i

    pos = find_error_position_by_syndrome(s, matrices)

    corrected = received.copy()
    if pos is not None:
        print(f"[Dec] Исправляем код: инвертируем бит в позиции {pos + 1}.")
        corrected[pos] ^= 1
        print(f"      Исправленный код c': {bits_to_str(corrected)}")
    else:
        print("[Dec] Коррекция не выполняется (ошибок нет или ошибка некорректируемая).")

    had_err = not np.all(s == 0)
    return DecodeResult(
        received=received,
        corrected=corrected,
        syndrome=s,
        syndrome_int=syndrome_int,
        detected_error_pos=pos,
        had_error=had_err,
    )


def extract_information_bits(codeword: np.ndarray, k: int) -> np.ndarray:
    """
    Для систематического кода вида [a_1..a_k | b_1..b_p] первые k разрядов — информационные.
    """
    if codeword.shape[0] < k:
        raise ValueError("Длина кодовой комбинации меньше k.")
    return codeword[:k].copy()
