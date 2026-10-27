"""Score-distribution drift used by the offline evaluation and the live service."""
import numpy as np


def psi(reference, current):
    """Population stability index over fixed [0, 1] probability bins with smoothed counts."""
    edges = np.linspace(0, 1, 11)
    a = np.histogram(reference, edges)[0] + .5
    b = np.histogram(current, edges)[0] + .5
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))
