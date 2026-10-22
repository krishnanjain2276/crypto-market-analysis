"""Gaussian Hidden Markov Model written directly in NumPy (no hmmlearn).

Diagonal-covariance Gaussian emissions, Baum-Welch EM with the scaled forward-backward
recursions from Rabiner (1989) so long samples don't underflow, a few random restarts, and
Viterbi decoding. States are sorted by the mean of the first feature after fitting, so state 0
is always the weakest-return regime and labels don't swap between refits.

Careful with the two kinds of probability:
  filter_proba  = P(state_t | data up to t)    causal, fine to trade on
  predict_proba = P(state_t | all the data)    uses the future, only for describing history
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HMMResult:
    log_likelihood: float
    n_iter: int
    converged: bool


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6,
                 n_init: int = 5, var_floor: float = 1e-4, seed: int = 0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.var_floor = var_floor
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _as2d(x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[:, None] if x.ndim == 1 else x

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.loc_) / self.scale_

    def _log_emission(self, z: np.ndarray) -> np.ndarray:
        """log N(z_t | mu_k, diag(var_k)) for all t, k -> shape (T, K)."""
        diff = z[:, None, :] - self.mu_[None, :, :]
        return -0.5 * (np.log(2 * np.pi * self.var_[None]) + diff ** 2 / self.var_[None]).sum(-1)

    def _forward_backward(self, logb: np.ndarray):
        T, K = logb.shape
        shift = logb.max(axis=1, keepdims=True)
        b = np.exp(logb - shift)
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.pi_ * b[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        A = self.A_
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * b[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_sum = A * (alpha[:-1].T @ (b[1:] * beta[1:] / c[1:, None]))
        loglik = float(np.log(c).sum() + shift.sum())
        return alpha, gamma, xi_sum, loglik

    def _init_params(self, z: np.ndarray, attempt: int):
        T, D = z.shape
        K = self.K
        if attempt == 0:  # deterministic: split the sample by quantiles of feature 0
            order = np.argsort(z[:, 0])
            chunks = np.array_split(order, K)
            self.mu_ = np.array([z[ch].mean(0) for ch in chunks])
            self.var_ = np.array([z[ch].var(0) + self.var_floor for ch in chunks])
        else:
            idx = self.rng.choice(T, K, replace=False)
            self.mu_ = z[idx] + self.rng.normal(0, 0.1, (K, D))
            self.var_ = np.ones((K, D)) * self.rng.uniform(0.5, 1.5, (K, 1))
        self.pi_ = np.full(K, 1.0 / K)
        self.A_ = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(self.A_, 0.95)

    def _em(self, z: np.ndarray) -> HMMResult:
        prev = -np.inf
        converged = False
        for it in range(1, self.n_iter + 1):  # noqa: B007 (it is reported after the loop)
            _, gamma, xi_sum, ll = self._forward_backward(self._log_emission(z))
            # M-step
            self.pi_ = gamma[0] + 1e-12
            self.pi_ /= self.pi_.sum()
            self.A_ = xi_sum + 1e-12
            self.A_ /= self.A_.sum(axis=1, keepdims=True)
            w = gamma.sum(0)[:, None]
            self.mu_ = gamma.T @ z / w
            diff2 = (z[:, None, :] - self.mu_[None]) ** 2
            self.var_ = (gamma[:, :, None] * diff2).sum(0) / w + self.var_floor
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                converged = True
                break
            prev = ll
        return HMMResult(ll, it, converged)

    def fit(self, x) -> GaussianHMM:
        x = self._as2d(x)
        x = x[np.isfinite(x).all(1)]
        self.loc_, self.scale_ = x.mean(0), x.std(0) + 1e-12
        z = self._scale(x)
        best = None
        for attempt in range(self.n_init):
            self._init_params(z, attempt)
            res = self._em(z)
            if best is None or res.log_likelihood > best[0].log_likelihood:
                best = (res, self.pi_.copy(), self.A_.copy(), self.mu_.copy(), self.var_.copy())
        self.result_, self.pi_, self.A_, self.mu_, self.var_ = best
        self._sort_states()
        self.n_obs_, self.n_features_ = z.shape
        return self

    def _sort_states(self):
        order = np.argsort(self.mu_[:, 0])
        self.mu_, self.var_, self.pi_ = self.mu_[order], self.var_[order], self.pi_[order]
        self.A_ = self.A_[np.ix_(order, order)]

    def filter_proba(self, x) -> np.ndarray:
        """Causal state probabilities P(s_t | x_1..x_t)."""
        alpha, *_ = self._forward_backward(self._log_emission(self._scale(self._as2d(x))))
        return alpha

    def predict_proba(self, x) -> np.ndarray:
        """Smoothed state probabilities P(s_t | x_1..x_T) (uses future data)."""
        _, gamma, _, _ = self._forward_backward(self._log_emission(self._scale(self._as2d(x))))
        return gamma

    def viterbi(self, x) -> np.ndarray:
        """Most likely state sequence (log-space dynamic programming)."""
        logb = self._log_emission(self._scale(self._as2d(x)))
        T, K = logb.shape
        logA = np.log(self.A_)
        delta = np.log(self.pi_) + logb[0]
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + logA
            psi[t] = scores.argmax(0)
            delta = scores.max(0) + logb[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta.argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score(self, x) -> float:
        return self._forward_backward(self._log_emission(self._scale(self._as2d(x))))[3]

    @property
    def means(self) -> np.ndarray:
        """State means in the original (unscaled) feature units."""
        return self.mu_ * self.scale_ + self.loc_

    @property
    def stds(self) -> np.ndarray:
        return np.sqrt(self.var_) * self.scale_

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected time spent in each state per visit: 1 / (1 - a_kk)."""
        return 1.0 / (1.0 - np.diag(self.A_))

    def bic(self, x) -> float:
        x = self._as2d(x)
        K, D = self.K, x.shape[1]
        n_params = (K - 1) + K * (K - 1) + 2 * K * D
        return -2 * self.score(x) + n_params * np.log(len(x))


def select_n_states(x, candidates=(2, 3, 4), seed: int = 0) -> tuple[int, dict[int, float]]:
    bics = {k: GaussianHMM(k, seed=seed).fit(x).bic(x) for k in candidates}
    return min(bics, key=bics.get), bics
