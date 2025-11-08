from __future__ import annotations

from pathlib import Path
import numpy as np


def save_csv(dirpath: Path, name: str, arr: np.ndarray) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"{name}.csv"
    np.savetxt(p, np.asarray(arr, dtype=float), delimiter=",", fmt="%.10f")
    print(f"[Сохр] Файл сохранён: {p}")
