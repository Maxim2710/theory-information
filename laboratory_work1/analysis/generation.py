from __future__ import annotations

import numpy as np
from typing import List

def make_probability_array(n: int, rng: np.random.Generator) -> List[float]:
    """
    Генерирует массив вероятностей длиной n.
    Используем распределение Дирихле с параметрами (1,1,...,1),
    чтобы гарантировать p_i > 0 и Σ p_i = 1.
    """
    probs = rng.dirichlet(np.ones(n)).tolist()
    return probs
