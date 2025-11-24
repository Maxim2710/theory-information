from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import contextlib
import sys

import numpy as np

from analysis.parameters import (
    CodeParameters,
    make_code_parameters,
    print_parameters_info,
)
from analysis.matrices import build_matrices
from analysis.experiments import run_experiments


DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6
VARIANT_MIN = 1
VARIANT_MAX = 54


@dataclass
class Tee:
    """
    Дублирует stdout в два потока (консоль + файл).
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
    print("=== ЛАБОРАТОРНАЯ РАБОТА № 4 — «СИСТЕМАТИЧЕСКИЙ КОД» ===")
    print("Цель работы: построить помехоустойчивый систематический код, позволяющий\n"
          "обнаруживать и исправлять все однократные ошибки (t = 1, d_min ≥ 3).\n")
    print("Структура программы:")
    print("  • Этап 1 — определение параметров кода (k, p, n) по номеру варианта,")
    print("             проверка формул (4.1) и (4.2).")
    print("  • Этап 2 — построение производящей матрицы P_{n,k} и проверочной матрицы H.")
    print("  • Этап 3 — численные эксперименты (Задание II) с подробными логами хода работы.\n")

    # 1) Ввод номера варианта и числа экспериментов
    variant = _ask_int(
        f"Введите номер варианта ({VARIANT_MIN}–{VARIANT_MAX}) [по умолчанию {DEFAULT_VARIANT}]: ",
        default=DEFAULT_VARIANT,
        lo=VARIANT_MIN,
        hi=VARIANT_MAX,
    )
    experiments = _ask_int(
        f"Сколько экспериментов выполнить? (не меньше {MIN_EXPERIMENTS}) "
        f"[по умолчанию {MIN_EXPERIMENTS}]: ",
        default=MIN_EXPERIMENTS,
        lo=MIN_EXPERIMENTS,
        hi=None,
    )

    # 2) Подготовка директории для отчёта
    root = Path(__file__).resolve().parent
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"lr4_report_{ts}_v{variant}_{experiments}exp.txt"

    # 3) Организация Tee: вывод в консоль и в файл-отчёт
    with report_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.__stdout__, f)
        with contextlib.redirect_stdout(tee):
            print(f"[Report] Файл отчёта: {report_path}")
            print(f"[Report] Дата/время запуска: {datetime.now().isoformat(timespec='seconds')}\n")

            # 4) Параметры кода (Задание, формулы (4.1), (4.2))
            params: CodeParameters = make_code_parameters(variant=variant, d_min=3)
            print_parameters_info(params)

            # 5) Построение матриц G и H (Задание I)
            print("\n=== ЗАДАНИЕ I. ПОСТРОЕНИЕ ПРОИЗВОДЯЩЕЙ И ПРОВЕРОЧНОЙ МАТРИЦ ===")
            print("[Step] Строим матрицы P_{n,k} (производящая матрица G) и H по формулам (4.4)–(4.8).")
            matrices = build_matrices(params)

            # 6) Инициализация ГСЧ (как в предыдущих работах — случайный seed из ОС)
            ss = np.random.SeedSequence()
            rng = np.random.default_rng(ss)
            shown_seed = int(ss.generate_state(1)[0])
            print("\n[Init RNG] Инициализация генератора случайных чисел:")
            print(f"          Использован seed от ОС: OS({shown_seed})")
            print("          (для воспроизводимости экспериментов можно зафиксировать seed явно).\n")

            # 7) Запуск серии экспериментов (Задание II и III)
            run_experiments(
                params=params,
                matrices=matrices,
                experiments=experiments,
                rng=rng,
                report_dir=report_dir,
            )

    # Финальное сообщение только в консоль
    print(f"\n[OK] Лог лабораторной работы №4 сохранён в файле: {report_path}")


if __name__ == "__main__":
    main()
