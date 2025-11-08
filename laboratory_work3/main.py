from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path
import sys
import contextlib
import numpy as np

from analysis.experiment import run_series

DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6
VARIANT_MIN = 1
VARIANT_MAX = 54


def variant_to_N(variant: int) -> int:
    return 7 + variant


@dataclass
class Tee:
    a: any
    b: any
    def write(self, data: str) -> None:
        self.a.write(data); self.b.write(data)
    def flush(self) -> None:
        self.a.flush(); self.b.flush()


def parse_args() -> argparse.Namespace:
    """
    Флаги оставлены для seed / запрета CSV / явного лог-файла.
    Вариант и число экспериментов спрашиваем интерактивно.
    """
    p = argparse.ArgumentParser(description="ЛР №3 — Обобщенные характеристики сигналов и каналов")
    p.add_argument("--seed", type=int, default=None, help="фиксированный seed (по умолчанию — OS entropy)")
    p.add_argument("--no-csv", action="store_true", help="не сохранять CSV")
    p.add_argument("--logfile", type=str, default=None, help="путь к текстовому лог-файлу")
    return p.parse_args()


def _ask_int(prompt: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    raw = input(prompt).strip()
    if not raw:
        val = default
    else:
        try:
            val = int(raw)
        except Exception:
            val = default
    if lo is not None and val < lo:
        print(f"[Внимание] Значение ниже {lo}; применено {lo}."); val = lo
    if hi is not None and val > hi:
        print(f"[Внимание] Значение выше {hi}; применено {hi}."); val = hi
    return val


def main() -> None:
    args = parse_args()

    # интерактивные вопросы с дефолтами 16 и 6
    variant = _ask_int(
        f"Введите номер варианта ({VARIANT_MIN}–{VARIANT_MAX}) [по умолчанию {DEFAULT_VARIANT}]: ",
        default=DEFAULT_VARIANT, lo=VARIANT_MIN, hi=VARIANT_MAX
    )
    experiments = _ask_int(
        f"Сколько экспериментов выполнить? (не меньше {MIN_EXPERIMENTS}) [по умолчанию {MIN_EXPERIMENTS}]: ",
        default=MIN_EXPERIMENTS, lo=MIN_EXPERIMENTS
    )

    N = variant_to_N(variant)

    series_root = Path(__file__).resolve().parent / "report"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    series_dir = series_root / f"series_{ts}_v{variant}_N{N}"

    # RNG
    if args.seed is None:
        ss = np.random.SeedSequence()
        rng = np.random.default_rng(ss)
        shown_seed = int(ss.generate_state(1)[0])
        seed_msg = f"OS({shown_seed})"
    else:
        rng = np.random.default_rng(args.seed)
        seed_msg = str(args.seed)

    header = [
        "ЛАБОРАТОРНАЯ РАБОТА №3 — ОБОБЩЕННЫЕ ХАРАКТЕРИСТИКИ СИГНАЛОВ И КАНАЛОВ",
        f"Вариант: {variant} → N = {N} (log2(N) = {log2(N):.6f} бит)",
        f"Экспериментов: {experiments}",
        f"Инициализация ГСЧ: {seed_msg}",
        f"q = 1/(2N) = {1.0/(2*N):.6f}",
        "-" * 78,
        "Ход работы: (a) P(Y); (b) τ_i; (c) P(Y|Z); (d) без помех; (e) с помехами",
        "-" * 78,
    ]
    header_text = "\n".join(header)

    if args.logfile:
        log_path = Path(args.logfile).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            tee = Tee(sys.__stdout__, f)
            with contextlib.redirect_stdout(tee):
                print(header_text)
                run_series(N=N, experiments=experiments, rng=rng, series_dir=series_dir, save_csv=not args.no_csv)
        print(f"\n[OK] Лог сохранён: {log_path}")
        if not args.no_csv:
            print(f"[OK] CSV сохранены в каталоге: {series_dir}")
    else:
        print(header_text)
        run_series(N=N, experiments=experiments, rng=rng, series_dir=series_dir, save_csv=not args.no_csv)
        if not args.no_csv:
            print(f"\n[OK] CSV сохранены в каталоге: {series_dir}")


if __name__ == "__main__":
    main()
