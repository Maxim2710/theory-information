from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .parameters import HammingParameters
from .layout import HammingLayout, build_layout, print_layout
from .encode_decode import (
    generate_information_word,
    encode_hamming,
    add_global_parity,
    correct_code,
    extract_information_bits,
    bits_to_str,
)
from .channel import transmit_through_channel, ChannelResult


@dataclass
class ExperimentSummary:
    true_weight: int
    detected_weight: int
    correction_success: bool
    double_detected: bool


def _fmt_bits(arr: np.ndarray) -> str:
    return bits_to_str(arr)


def run_experiments(params: HammingParameters,
                    experiments: int,
                    rng: np.random.Generator,
                    report_dir: Path) -> List[ExperimentSummary]:
    """
    Основной цикл ЛР №5 (Задание I и II).

    Для каждого эксперимента:
      a) генерируем случайную информационную комбинацию длины k;
      b) строим код Хэмминга (исправление однократных ошибок);
      c) добавляем дополнительный проверочный разряд (обнаружение двукратных ошибок);
      d) моделируем передачу по каналу с кратностью ошибки 0..2;
      e) на приёмнике рассчитываем синдром, определяем кратность и позицию ошибки (для однократных);
      f) выполняем коррекцию (если возможно) и проверяем совпадение информационной части.
    """
    if experiments < 6:
        experiments = 6

    print("\n=== ЗАДАНИЕ I. КОМПЛЕКС ЧИСЛЕННЫХ ЭКСПЕРИМЕНТОВ (ЛР №5) ===")
    print(f"[Plan] Число экспериментов: {experiments} (не менее 6).")

    layout: HammingLayout = build_layout(params)
    print_layout(layout)

    summaries: List[ExperimentSummary] = []

    total = experiments
    count_w0 = 0
    count_w1 = 0
    count_w2 = 0
    corrected_single = 0
    detected_double = 0

    for exp_no in range(1, experiments + 1):
        print("\n" + "=" * 78)
        print(f"ЭКСПЕРИМЕНТ {exp_no}")
        print("=" * 78)

        # (a) Генерация информационной комбинации.
        print("\n(a) Генерация случайной информационной кодовой комбинации (k двоичных разрядов).")
        info = generate_information_word(params.k, rng)
        print(f"    Информационная комбинация a (k = {params.k}): {bits_to_str(info)}")

        # (b) Построение кода Хэмминга.
        print("\n(b) Построение кода Хэмминга, позволяющего исправлять однократные ошибки.")
        code_h = encode_hamming(info, params, layout)

        # (c) Добавление дополнительного проверочного разряда.
        print("\n(c) Модификация кода Хэмминга для обнаружения двукратных ошибок (добавление S_{p+1}).")
        code_ext = add_global_parity(code_h, params)

        # (d) Передача по каналу с кратностью ошибки 0..2.
        print("\n(d) Передача сформированного кода по каналу с помехами (кратность ошибки 0..2).")
        chan_res: ChannelResult = transmit_through_channel(code_ext, params, rng)
        true_weight = chan_res.error_weight
        if true_weight == 0:
            count_w0 += 1
        elif true_weight == 1:
            count_w1 += 1
        elif true_weight == 2:
            count_w2 += 1

        # (e) Обработка на приёмной стороне: синдром, кратность, позиция, коррекция.
        print("\n(e) Обработка принятого кода на приёмной стороне: синдром, кратность, позиция ошибки, коррекция.")
        dec_res = correct_code(chan_res.received, params, layout)

        detected_weight = dec_res.error.weight
        is_double = (dec_res.error.weight == 2)

        # Проверка успешности восстановления информационной части.
        info_decoded = extract_information_bits(dec_res.corrected, layout)
        ok_info = np.array_equal(info_decoded, info)

        print("\n[Check] Сравнение информационной части до и после передачи/декодирования.")
        print(f"  Исходная информационная комбинация a:  {bits_to_str(info)}")
        print(f"  Восстановленная информационная комбинация a': {bits_to_str(info_decoded)}")
        print(f"  Совпадение a == a'? → {'ДА' if ok_info else 'НЕТ'}")

        if true_weight == 1 and ok_info:
            corrected_single += 1
        if true_weight == 2 and is_double:
            detected_double += 1

        summaries.append(
            ExperimentSummary(
                true_weight=true_weight,
                detected_weight=detected_weight,
                correction_success=ok_info,
                double_detected=is_double,
            )
        )

        print("\n[Summary per experiment]")
        print(f"  Истинная кратность ошибки: {true_weight}")
        print(f"  Оценённая по синдрому кратность: {detected_weight}")
        print(f"  Коррекция информационной части успешна? → {'ДА' if ok_info else 'НЕТ'}")

    # Итоги по серии
    print("\n=== ИТОГИ СЕРИИ ЭКСПЕРИМЕНТОВ (ЗАДАНИЕ I) ===")
    print(f"[Stat] Всего экспериментов: {total}")
    print(f"[Stat] Без ошибок (кратность 0): {count_w0}")
    print(f"[Stat] Однократные ошибки (кратность 1): {count_w1}")
    print(f"[Stat] Двукратные ошибки (кратность 2): {count_w2}")
    print(f"[Stat] Успешно исправлено однократных ошибок: {corrected_single}/{count_w1 if count_w1 > 0 else 1}")
    print(f"[Stat] Двукратные ошибки корректно обнаружены как двукратные: "
          f"{detected_double}/{count_w2 if count_w2 > 0 else 1}")

    print("\n=== ЗАДАНИЕ II. ВЫВОДЫ ПО РАБОТЕ ===")
    print("1) Базовый код Хэмминга (d_min = 3) обеспечивает исправление всех однократных ошибок.")
    print("2) Добавление дополнительного проверочного разряда повышает минимальное кодовое расстояние до d_min = 4,")
    print("   что позволяет обнаруживать двукратные ошибки (по значению S_(p+1)).")
    print("3) В экспериментах однократные ошибки успешно обнаруживаются по синдрому S1..Sp и исправляются,")
    print("   что подтверждается совпадением информационной части a и a' после декодирования.")
    print("4) Двукратные ошибки приводят к ненулевому синдрому при S_(p+1) = 0, что соответствует модели")
    print("   «обнаружение без исправления» для таких ошибок.")
    print("5) При отсутствии ошибок синдром равен нулю и расширенный код проходит проверку без корректировок.")

    return summaries
