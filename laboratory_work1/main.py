from __future__ import annotations

from math import log2
from datetime import datetime
from pathlib import Path
import sys
import contextlib

from analysis.experiments import run_experiments

DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6
VARIANT_MIN = 1
VARIANT_MAX = 54


def variant_to_N(variant: int) -> int:
    return variant + 7


def ask_int(prompt: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    if min_value is not None and val < min_value:
        print(f"[Warn] Значение < {min_value}; установлено {min_value}.")
        val = min_value
    if max_value is not None and val > max_value:
        print(f"[Warn] Значение > {max_value}; установлено {max_value}.")
        val = max_value
    return val


class Tee:
    """Пишет в несколько потоков сразу (консоль + файл)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def main() -> None:
    # Ввод параметров
    variant = ask_int(
        prompt=f"Введите номер варианта ({VARIANT_MIN}–{VARIANT_MAX}) [по умолчанию {DEFAULT_VARIANT}]: ",
        default=DEFAULT_VARIANT, min_value=VARIANT_MIN, max_value=VARIANT_MAX
    )
    N = variant_to_N(variant)

    experiments = ask_int(
        prompt=f"Сколько экспериментов выполнить? (не меньше {MIN_EXPERIMENTS}) [по умолчанию {MIN_EXPERIMENTS}]: ",
        default=MIN_EXPERIMENTS, min_value=MIN_EXPERIMENTS
    )

    # Путь к файлу отчёта: <корень>/laboratory_work1/report/...
    project_root = Path(__file__).resolve().parent  # это папка laboratory_work1
    report_dir = project_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{ts}_v{variant}_N{N}_{experiments}exp.txt"

    # Тиражируем вывод: в консоль и в файл
    with report_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.__stdout__, f)
        with contextlib.redirect_stdout(tee):
            print(f"[Report] Файл отчёта: {report_path}")
            print(f"[Report] Дата/время: {datetime.now().isoformat(timespec='seconds')}")
            print("=== ЛР №1: Количество информации и неопределенность сообщения ===")
            print(f"[Init] Вариант: {variant} -> размер алфавита N = {N}")
            h_max = log2(N)
            print(f"[Init] Максимальная энтропия H_max = log2(N) = {h_max:.5f} бит")

            # seed=None -> ОС-энтропия (каждый запуск уникален)
            run_experiments(N=N, experiments=experiments, seed=None, print_hmax=h_max)

    # Финальное уведомление в консоль (на всякий случай)
    print(f"\n[OK] Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
