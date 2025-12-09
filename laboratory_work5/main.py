from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import contextlib
import sys

import numpy as np

from analysis.parameters import (
    make_hamming_parameters,
    print_parameters_info,
    MIN_VARIANT,
    MAX_VARIANT,
)
from analysis.experiments import run_experiments


DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6


@dataclass
class Tee:
    """
    Дублирует stdout в консоль и в файл отчёта.
    """
    a: any
    b: any

    def write(self, data: str) -> None:
        self.a.write(data)
        self.b.write(data)

    def flush(self) -> None:
        self.a.flush()
        self.b.flush()


def _ask_int(prompt: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    """
    Безопасный ввод целого числа с дефолтным значением и ограничениями.
    Пустой ввод → default.
    """
    raw = input(prompt).strip()
    if not raw:
        val = default
    else:
        try:
            val = int(raw)
        except Exception:
            val = default

    if lo is not None and val < lo:
        print(f"[Внимание] Значение ниже {lo}; установлено {lo}.")
        val = lo
    if hi is not None and val > hi:
        print(f"[Внимание] Значение выше {hi}; установлено {hi}.")
        val = hi
    return val


def main() -> None:
    print("=== ЛАБОРАТОРНАЯ РАБОТА № 5 — «КОД ХЭММИНГА» ===")
    print("Цель работы: построить помехоустойчивый код (код Хэмминга), позволяющий\n"
          "обнаруживать и исправлять однократные ошибки, а также обнаруживать двукратные.\n")
    print("Ход работы соответствует методическим указаниям:")
    print(" • построение кода Хэмминга по заданному k (таблица вариантов),")
    print(" • модификация кода для обнаружения двукратных ошибок,")
    print(" • моделирование передачи по каналу с кратностью ошибок 0..2,")
    print(" • вычисление синдрома, определение кратности и позиции ошибки, коррекция кода.\n")

    variant = _ask_int(
        prompt=f"Введите номер варианта ({MIN_VARIANT}–{MAX_VARIANT}) "
               f"[по умолчанию {DEFAULT_VARIANT}]: ",
        default=DEFAULT_VARIANT,
        lo=MIN_VARIANT,
        hi=MAX_VARIANT,
    )

    experiments = _ask_int(
        prompt=f"Сколько экспериментов выполнить? (не меньше {MIN_EXPERIMENTS}) "
               f"[по умолчанию {MIN_EXPERIMENTS}]: ",
        default=MIN_EXPERIMENTS,
        lo=MIN_EXPERIMENTS,
        hi=None,
    )

    # Подготовка каталога для отчёта.
    project_root = Path(__file__).resolve().parent
    report_dir = project_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"lr5_report_{ts}_v{variant}_k{variant + 7}_{experiments}exp.txt"

    with report_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.__stdout__, f)
        with contextlib.redirect_stdout(tee):
            print(f"[Report] Файл отчёта: {report_path}")
            print(f"[Report] Дата/время запуска: {datetime.now().isoformat(timespec='seconds')}\n")

            # Этап 1: параметры кода.
            params = make_hamming_parameters(variant)
            print_parameters_info(params)

            # Инициализация ГСЧ (как в предыдущих работах: SeedSequence от ОС).
            ss = np.random.SeedSequence()
            rng = np.random.default_rng(ss)
            shown_seed = int(ss.generate_state(1)[0])
            print("\n[Init RNG] Инициализация генератора случайных чисел.")
            print(f"  Использован seed от ОС: OS({shown_seed})")
            print("  Для воспроизводимости можно зафиксировать seed явно при необходимости.\n")

            # Запуск серии экспериментов (Задание I и II).
            run_experiments(params=params, experiments=experiments, rng=rng, report_dir=report_dir)

            print(f"\n[OK] Отчёт по ЛР №5 сохранён: {report_path}")

    # Финальное уведомление только в консоль.
    print(f"\n[OK] Лог лабораторной работы №5 записан в файл: {report_path}")


if __name__ == "__main__":
    main()
