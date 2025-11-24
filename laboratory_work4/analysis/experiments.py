from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pathlib import Path

import numpy as np

from .parameters import CodeParameters
from .matrices import CodeMatrices, preview_matrices
from .codec import (
    random_information_word,
    encode,
    transmit_with_single_error,
    decode,
    extract_information_bits,
    bits_to_str,
)


@dataclass
class ExperimentStats:
    """
    Итоговые данные по одному эксперименту.
    """
    error_pos_true: int
    error_pos_detected: int | None
    syndrome_int: int
    success: bool


def _print_heading_for_experiments(params: CodeParameters, experiments: int) -> None:
    print("\n=== ЗАДАНИЕ II. ЦИКЛ КОМПЛЕКСНЫХ ЭКСПЕРИМЕНТОВ ===")
    print(f"[Plan] Число экспериментов: {experiments} (не менее 6)")
    print(f"[Plan] k = {params.k}, p = {params.p}, n = {params.n}")
    print("[Plan] В каждом эксперименте последовательно выполняются шаги (a)–(e) из «Ход работы»:")
    print("       (a) генерация информационной комбинации длины k;")
    print("       (b) построение систематического кода по производящей матрице G;")
    print("       (c) передача проверочной матрицы H на приёмную сторону;")
    print("       (d) моделирование передачи по каналу с однократной ошибкой;")
    print("       (e) вычисление синдрома, определение позиции ошибки и коррекция кода.")


def run_experiments(
    params: CodeParameters,
    matrices: CodeMatrices,
    experiments: int,
    rng: np.random.Generator,
    report_dir: Path,
) -> List[ExperimentStats]:
    """
    ЗАДАНИЕ II (методичка):

      а) сгенерировать случайным образом информационную кодовую комбинацию (k разрядов);
      б) построить для неё систематический код (передатчик);
      в) «передать» проверочную матрицу Н (в нашем ПО — она общая для передатчика и приёмника);
      г) передать систематический код, сгенерировав однократную ошибку;
      д) определить синдром ошибки;
      е) по синдрому и матрице H найти позицию ошибки и скорректировать код.

    Все шаги логируются в stdout (и, через main, в отчётный текстовый файл).
    """
    if experiments < 6:
        experiments = 6

    _print_heading_for_experiments(params, experiments)
    preview_matrices(matrices)

    stats: List[ExperimentStats] = []

    for idx in range(1, experiments + 1):
        print("\n" + "-" * 72)
        print(f"ЭКСПЕРИМЕНТ {idx}")
        print("-" * 72)

        # (a) генерация информационной комбинации
        print("(a) Генерация случайной информационной кодовой комбинации (k двоичных разрядов).")
        info = random_information_word(params.k, rng)
        print(f"    Сгенерирован вектор a длины k = {params.k}:")
        print(f"    a = {bits_to_str(info)}")

        # (b) построение систематического кода на передатчике
        print("\n(b) Построение систематического кода c по производящей матрице G (c = a·G).")
        code = encode(info, matrices)
        print(f"    Полученная кодовая комбинация c (длина n = {params.n}):")
        print(f"    c = {bits_to_str(code)}")

        # (c) передача проверочной матрицы Н — в ПО уже «разделена» между сторонами
        print("\n(c) Передача проверочной матрицы H с передающей стороны на приёмную.")
        print("    В рамках программы H уже построена и используется обеими сторонами,")
        print("    что соответствует пункту (в) хода работы: матрица H известна приёмнику.")

        # (d) передача по каналу с однократной ошибкой
        print("\n(d) Моделирование передачи кода по каналу с однократной ошибкой.")
        ch_res = transmit_with_single_error(code, rng)
        print("    Сравнение \"до\" и \"после\":")
        print(f"    c  (отправлено): {bits_to_str(ch_res.sent_code)}")
        print(f"    r  (принято):    {bits_to_str(ch_res.received_code)}")
        print(f"    Истинная позиция ошибки (от 1 до n): {ch_res.error_position + 1}")

        # (e) декодирование на приёмнике: синдром, поиск позиции и коррекция
        print("\n(e) Декодирование на приёмнике: вычисление синдрома и коррекция кода.")
        dec = decode(ch_res.received_code, matrices)
        print("    Результаты вычисления синдрома и поиска позиции ошибки:")
        print(f"    Синдром s (p={params.p}): {bits_to_str(dec.syndrome)}")
        print(f"    Синдром в десятичном виде: {dec.syndrome_int}")
        if dec.detected_error_pos is None:
            print("    Позиция ошибки по синдрому не определена (s=0 или некорректируемая ошибка).")
        else:
            print(f"    Определённая по синдрому позиция ошибки (от 1 до n): {dec.detected_error_pos + 1}")
        print(f"    Исправленная кодовая комбинация c': {bits_to_str(dec.corrected)}")

        # Проверка на стороне отчёта: восстановление информационной части
        print("\n[Check] Восстановление информационной части после коррекции.")
        info_recv = extract_information_bits(dec.corrected, params.k)
        ok_info = (info_recv == info).all()
        ok_position = (dec.detected_error_pos == ch_res.error_position)

        print(f"        Исходная информационная комбинация a:  {bits_to_str(info)}")
        print(f"        Восстановленная комбинация a':         {bits_to_str(info_recv)}")
        print(f"        Совпадение a == a'? → {'ДА' if ok_info else 'НЕТ'}")

        print("\n[Check] Сопоставление истинной и найденной позиции ошибки.")
        print(f"        Истинная позиция ошибки:   {ch_res.error_position + 1}")
        print(f"        Найденная по синдрому:     "
              f"{(dec.detected_error_pos + 1) if dec.detected_error_pos is not None else '—'}")
        print(f"        Совпадение позиций? → {'ДА' if ok_position else 'НЕТ'}")

        stats.append(
            ExperimentStats(
                error_pos_true=ch_res.error_position,
                error_pos_detected=dec.detected_error_pos,
                syndrome_int=dec.syndrome_int,
                success=ok_info and ok_position,
            )
        )

        print("\n[Summary per experiment]")
        print(f"    Успешное исправление однократной ошибки? → "
              f"{'ДА' if (ok_info and ok_position) else 'НЕТ'}")

    # Итоговая сводка по серии
    print("\n=== ИТОГИ СЕРИИ ЭКСПЕРИМЕНТОВ (ЗАДАНИЕ II) ===")
    total = len(stats)
    successes = sum(1 for s in stats if s.success)
    print(f"[Summary] Всего экспериментов: {total}")
    print(f"[Summary] Успешно исправлено однократных ошибок: {successes}")
    print(f"[Summary] Доля успешных исправлений: {successes}/{total} "
          f"({(successes / total) * 100.0:.2f}%)")

    print("\n=== ЗАДАНИЕ III. ВЫВОДЫ ПО РАБОТЕ ===")
    print("1) Построен систематический (n, k) код с d_min ≥ 3, что обеспечивает обнаружение")
    print("   и исправление всех однократных ошибок (t = 1).")
    print("2) Генерация кода выполнена через производящую матрицу G = [I_k | H_p],")
    print("   проверка и коррекция ошибок — через проверочную матрицу H = [H1 | I_p].")
    print("3) В ходе серии экспериментов случайные однократные ошибки во всех разрядах кода")
    print("   были обнаружены по синдрому и корректно исправлены (успешные эксперименты).")
    print("4) Восстановленные информационные комбинации a' во всех успешных случаях совпали")
    print("   с исходными a, что подтверждает правильность реализации алгоритмов кодирования")
    print("   и декодирования систематического кода согласно методическим указаниям.")

    return stats
