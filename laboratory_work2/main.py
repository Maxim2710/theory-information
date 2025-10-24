from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path
import contextlib
import sys
import numpy as np

from analysis.experiments import run_experiments


DEFAULT_VARIANT = 16
MIN_EXPERIMENTS = 6
VARIANT_MIN = 1
VARIANT_MAX = 54


def variant_to_N_lab2(variant: int) -> int:
    """
    Соответствие варианта и N для ЛР №2.
    Требуется: вариант 16 → N = 23 → формула N = 7 + variant.
    """
    return 7 + variant


@dataclass
class Tee:
    """Дублирует stdout в файл и консоль."""
    a: any
    b: any
    def write(self, data: str) -> None:
        self.a.write(data); self.b.write(data)
    def flush(self) -> None:
        self.a.flush(); self.b.flush()


def _ask_int(prompt: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except Exception:
        return default
    if lo is not None and v < lo:
        print(f"[Вним] Значение ниже {lo}; применено {lo}."); v = lo
    if hi is not None and v > hi:
        print(f"[Вним] Значение выше {hi}; применено {hi}."); v = hi
    return v


def main() -> None:
    print("=== ЛР №2: Количество информации при неполной достоверности сообщений ===")

    variant = _ask_int(
        f"Введите номер варианта ({VARIANT_MIN}–{VARIANT_MAX}) [по умолчанию {DEFAULT_VARIANT}]: ",
        default=DEFAULT_VARIANT, lo=VARIANT_MIN, hi=VARIANT_MAX
    )
    N = variant_to_N_lab2(variant)

    experiments = _ask_int(
        f"Сколько экспериментов выполнить? (не меньше {MIN_EXPERIMENTS}) [по умолчанию {MIN_EXPERIMENTS}]: ",
        default=MIN_EXPERIMENTS, lo=MIN_EXPERIMENTS
    )

    min_correct_prob = 0.70  # «не менее 70%»

    # Каталог отчёта
    root = Path(__file__).resolve().parent
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"lr2_report_{ts}_v{variant}_N{N}_{experiments}exp.txt"

    with report_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.__stdout__, f)
        with contextlib.redirect_stdout(tee):
            # RNG из ОС + печать seed-state для воспроизводимости
            ss = np.random.SeedSequence()
            rng = np.random.default_rng(ss)
            shown_seed = int(ss.generate_state(1)[0])

            print(f"[Отчёт] Файл отчёта: {report_path}")
            print(f"[Отчёт] Дата/время: {datetime.now().isoformat(timespec='seconds')}")
            print(f"[Иниц ] Вариант: {variant} → N = {N}, H_max = log2(N) = {log2(N):.6f} бит")
            print(f"[Иниц ] Инициализация ГСЧ = OS({shown_seed}); min p(x_j|y_j) = {min_correct_prob:.2f}")
            print("\n[Памят] Формулы методички: "
                  "(2.1) H(X); (2.3) P(X|Y); (2.4) p(Y); (2.6)–(2.7) P(X,Y); "
                  "(2.8) H(X|Y); (2.9) I(X,Y)=H(X)-H(X|Y).")

            series_dir = report_dir / f"series_{ts}_v{variant}_N{N}"
            stats = run_experiments(
                N=N,
                experiments=experiments,
                rng=rng,
                out_dir=series_dir,
                min_correct=min_correct_prob,
            )

            i_vals = [s.i_xy for s in stats]
            print("\n[Готово] Требования ЛР №2 выполнены.")
            print(f"[Готово] Среднее по серии Ī(X,Y) = {sum(i_vals)/len(i_vals):.6f} бит")
            print(f"[Готово] CSV сохранены в каталоге: {series_dir}")

    print(f"\n[ОК] Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
