from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import contextlib
import sys

from analysis.experiments import run_series


DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6
VARIANT_MIN = 1
VARIANT_MAX = 54


@dataclass
class Tee:
    a: any
    b: any
    def write(self, data: str) -> None:
        self.a.write(data); self.b.write(data)
    def flush(self) -> None:
        self.a.flush(); self.b.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ЛР №6 — Циклический код (исправление однократных ошибок)")
    p.add_argument("--seed", type=int, default=None, help="фиксированный seed (по умолчанию — OS entropy)")
    p.add_argument("--variant", type=int, default=DEFAULT_VARIANT, help="номер варианта (по умолчанию 16)")
    p.add_argument("--experiments", type=int, default=MIN_EXPERIMENTS, help="число экспериментов (>=6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variant = args.variant
    experiments = args.experiments
    if experiments < MIN_EXPERIMENTS:
        experiments = MIN_EXPERIMENTS
    if variant < VARIANT_MIN:
        variant = VARIANT_MIN
    if variant > VARIANT_MAX:
        variant = VARIANT_MAX

    root = Path(__file__).resolve().parent
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"lr6_report_{ts}_v{variant}_{experiments}exp.txt"

    with report_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.__stdout__, f)
        with contextlib.redirect_stdout(tee):
            print(f"[Report] Файл отчёта: {report_path}")
            print(f"[Report] Дата/время запуска: {datetime.now().isoformat(timespec='seconds')}")
            print(f"[Report] Вариант: {variant}, экспериментов: {experiments}")
            print("-" * 78)

            run_series(variant=variant, experiments=experiments, report_dir=report_dir, seed=args.seed)

    print(f"\n[OK] Лог лабораторной работы №6 сохранён: {report_path}")


if __name__ == "__main__":
    main()
