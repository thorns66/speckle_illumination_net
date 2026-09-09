"""Deterministic, GT-free 10-of-100 sensor-frame challenge selection."""
from __future__ import annotations

import numpy as np


def correlation_matrix(frames):
    x = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite sensor values')
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError('Zero spatial variance frame')
    x /= norms[:, None]
    return np.clip(x @ x.T, -1., 1.)


def score(matrix, indices):
    block = np.abs(matrix[np.ix_(indices, indices)])
    return float(block[np.triu_indices(len(indices), 1)].mean())


def local_search(matrix, initial, maximize=False):
    weights = np.abs(matrix).copy()
    np.fill_diagonal(weights, 0)
    chosen = np.sort(initial).astype(int)
    pairs = len(chosen) * (len(chosen) - 1) / 2
    sign = -1 if maximize else 1
    while True:
        outside = np.setdiff1d(np.arange(len(weights)), chosen)
        sums = weights[:, chosen].sum(axis=1)
        delta = (sums[outside][None, :] - weights[np.ix_(chosen, outside)]
                 - sums[chosen][:, None]) / pairs
        improvement = sign * delta
        best = float(improvement.min())
        if best >= -1e-12:
            return chosen, score(matrix, chosen)
        candidates = np.argwhere(improvement <= best + 1e-15)
        choices = []
        for i, j in candidates:
            new = chosen.copy()
            new[i] = outside[j]
            choices.append(tuple(sorted(new)))
        chosen = np.array(min(choices))


def select_extremes(matrix, *, seed=20260908, count=1000, starts=100):
    if matrix.shape != (100, 100):
        raise ValueError('Expected 100 frames')
    rng = np.random.default_rng(seed)
    candidates = set()
    while len(candidates) < count:
        candidates.add(tuple(sorted(rng.choice(100, 10, replace=False).tolist())))
    ranked = sorted((score(matrix, s), s) for s in candidates)
    result = {}
    for name, maximum in [('low', False), ('high', True)]:
        seeds = sorted(ranked, key=lambda p: ((-p[0] if maximum else p[0]), p[1]))[:starts]
        solutions = [local_search(matrix, s, maximum) for _, s in seeds]
        solution, value = min(solutions, key=lambda p: ((-p[1] if maximum else p[1]), tuple(p[0])))
        result[name] = {'input_indices': (solution + 1).tolist(), 'score': value,
                        'holdout_indices': (np.setdiff1d(np.arange(100), solution) + 1).tolist()}
    result['random_scores'] = [v for v, _ in ranked]
    result['random_indices'] = [[i + 1 for i in s] for _, s in ranked]
    return result
