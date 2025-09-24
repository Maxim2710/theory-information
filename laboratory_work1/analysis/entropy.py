from __future__ import annotations

from math import log2
from typing import Sequence

def private_entropy(p: float) -> float:
    """
    Частная энтропия (1.1): H(x_i) = log2(1 / p_i).
    Предполагается p > 0 (в нашем генераторе так и есть).
    """
    if p <= 0.0:
        raise ValueError("Вероятность должна быть > 0 для корректного вычисления энтропии.")
    return log2(1.0 / p)

def entropy(probabilities: Sequence[float]) -> float:
    """
    Энтропия совокупности (1.2): H(X) = - Σ p_i * log2(p_i)
    Нули игнорируем (в Dirichlet их не бывает).
    """
    h = 0.0
    for p in probabilities:
        h -= p * log2(p)
    return h

def information(probabilities: Sequence[float]) -> float:
    """
    Количество информации (1.4): I(X) = - Σ p_i * log2(p_i) ≡ H(X).
    Оставлено отдельной функцией для прозрачности соответствия методичке.
    """
    return entropy(probabilities)
