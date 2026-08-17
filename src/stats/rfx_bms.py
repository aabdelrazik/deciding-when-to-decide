"""Random-effects Bayesian model selection, and the protected exceedance probability.

Implements the variational scheme of Stephan et al. (2009) together with the
Bayesian omnibus risk and protected exceedance probability of Rigoux et al.
(2014). Written directly because neither SPM nor the VBA toolbox is available
here; validate_rfx_bms() below checks it against cases whose answers are known.

The model treats which model a subject uses as a random effect: subject i's
model m_i is drawn from a categorical distribution with population frequencies
r, and r itself carries a Dirichlet prior. The input is the subject-by-model log
evidence, for which -BIC/2 is the usual approximation.

Quantities returned:
    alpha  Dirichlet parameters of the posterior over r
    r      expected model frequencies, alpha / sum(alpha)
    xp     exceedance probability, P(r_k > r_j for all j != k)
    bor    Bayesian omnibus risk, the posterior probability that the model
           frequencies are in fact all equal, i.e. that the apparent
           differences are chance
    pxp    protected exceedance probability, xp * (1 - bor) + bor / K

PXP is the quantity to report. A large BOR drags every PXP toward 1/K, which is
what stops an exceedance probability near 1 from being read as evidence when the
data cannot actually distinguish the models.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, digamma, gammaln


def _dirichlet_expected_log(alpha: np.ndarray) -> np.ndarray:
    return digamma(alpha) - digamma(alpha.sum())


def _log_dirichlet_const(alpha: np.ndarray) -> float:
    return float(gammaln(alpha.sum()) - gammaln(alpha).sum())


def fit_rfx(log_evidence: np.ndarray, alpha0: float = 1.0,
            tol: float = 1e-8, max_iter: int = 10_000) -> dict:
    """Variational RFX model comparison (Stephan et al. 2009, eqs 8-12).

    Args:
        log_evidence: (N subjects, K models) log model evidence.
        alpha0: symmetric Dirichlet prior concentration.

    Returns:
        dict with alpha, r, g (per-subject model responsibilities), and the
        free energy F of the fitted model.
    """
    L = np.asarray(log_evidence, float)
    n, k = L.shape
    alpha = np.full(k, alpha0, float)

    for _ in range(max_iter):
        # responsibilities: subject-wise posterior over models
        u = L + _dirichlet_expected_log(alpha)
        u -= u.max(axis=1, keepdims=True)          # stabilise before exponentiating
        g = np.exp(u)
        g /= g.sum(axis=1, keepdims=True)

        alpha_new = alpha0 + g.sum(axis=0)
        if np.abs(alpha_new - alpha).max() < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return dict(alpha=alpha, r=alpha / alpha.sum(), g=g,
                F=_free_energy(L, alpha, g, alpha0))


def _free_energy(L: np.ndarray, alpha: np.ndarray, g: np.ndarray,
                 alpha0: float) -> float:
    """Negative free energy of the fitted RFX model."""
    k = alpha.size
    a0 = np.full(k, alpha0, float)
    elog = _dirichlet_expected_log(alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_g = np.where(g > 0, np.log(g), 0.0)
    return float(
        (g * L).sum()                               # expected log evidence
        + (g * elog[None, :]).sum()                 # expected log prior on m
        - (g * log_g).sum()                         # entropy of q(m)
        + _log_dirichlet_const(a0) - _log_dirichlet_const(alpha)
        + ((a0 - alpha) * elog).sum()               # KL term for q(r)
    )


def _null_free_energy(L: np.ndarray, alpha0: float = 1.0) -> float:
    """Free energy of the null in which every model is equally frequent.

    Under the null each subject's model is drawn uniformly, so the evidence for
    the family is the average of the per-subject evidences (Rigoux et al. 2014).
    """
    k = L.shape[1]
    m = L.max(axis=1, keepdims=True)
    return float((m.ravel() + np.log(np.exp(L - m).mean(axis=1))).sum())


def exceedance_probability(alpha: np.ndarray, n_samples: int = 200_000,
                           seed: int = 0) -> np.ndarray:
    """P(r_k > r_j for all j != k) under Dir(alpha).

    Two models have a closed form; beyond that this samples, which is what SPM
    does. 200k draws give a Monte Carlo error under 0.002.
    """
    k = alpha.size
    if k == 2:
        xp0 = float(np.exp(np.log(0.5) * 0))  # placeholder, replaced below
        # P(r1 > r2) for a Dirichlet(a1, a2) is the regularised incomplete beta
        from scipy.stats import beta as beta_dist
        p1 = 1.0 - beta_dist.cdf(0.5, alpha[0], alpha[1])
        return np.array([p1, 1.0 - p1])
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(alpha, size=n_samples)
    winners = draws.argmax(axis=1)
    return np.bincount(winners, minlength=k) / n_samples


def protected_exceedance_probability(log_evidence: np.ndarray,
                                     alpha0: float = 1.0,
                                     n_samples: int = 200_000,
                                     seed: int = 0) -> dict:
    """Full RFX comparison: frequencies, XP, BOR and PXP.

    Args:
        log_evidence: (N subjects, K models), e.g. -BIC/2.

    Returns:
        dict with alpha, r, xp, bor, pxp.
    """
    L = np.asarray(log_evidence, float)
    k = L.shape[1]
    fit = fit_rfx(L, alpha0=alpha0)
    xp = exceedance_probability(fit["alpha"], n_samples=n_samples, seed=seed)

    # BOR: posterior probability of the null, under equal prior odds
    f1, f0 = fit["F"], _null_free_energy(L, alpha0)
    bor = float(1.0 / (1.0 + np.exp(f1 - f0)))
    pxp = xp * (1.0 - bor) + bor / k
    return dict(alpha=fit["alpha"], r=fit["r"], xp=xp, bor=bor, pxp=pxp,
                F=f1, F_null=f0)


def validate_rfx_bms(verbose: bool = True) -> bool:
    """Check the implementation on cases whose answers are known.

    1. One model generates every subject, by a wide margin: that model should
       take essentially all the frequency, XP near 1, BOR near 0, PXP near 1.
    2. All models are equivalent: frequencies uniform, BOR near 1, and every
       PXP dragged to 1/K, which is the whole point of the protection.
    3. Two models, one favoured by a modest amount: the closed form for K=2 and
       the sampler must agree.
    """
    ok = True
    rng = np.random.default_rng(0)

    # 1. a clear winner
    n, k = 60, 4
    L = rng.normal(0, 1, (n, k))
    L[:, 1] += 20.0
    res = protected_exceedance_probability(L)
    good = res["pxp"][1] > 0.99 and res["bor"] < 0.01 and res["r"][1] > 0.9
    ok &= good
    if verbose:
        print(f"  1 clear winner   : r={res['r'].round(3)} pxp[1]={res['pxp'][1]:.4f} "
              f"BOR={res['bor']:.4f}  {'OK' if good else 'FAIL'}")

    # 2. nothing to choose between them
    L = rng.normal(0, 1e-6, (n, k))
    res = protected_exceedance_probability(L)
    good = res["bor"] > 0.9 and np.allclose(res["pxp"], 1 / k, atol=0.02)
    ok &= good
    if verbose:
        print(f"  2 null           : pxp={res['pxp'].round(3)} (1/K={1/k:.3f}) "
              f"BOR={res['bor']:.4f}  {'OK' if good else 'FAIL'}")

    # 3. K=2 closed form vs sampling
    # a weak effect, so the answer is strictly between 0.5 and 1 and the two
    # routes can actually disagree
    L = rng.normal(0, 1, (12, 2))
    L[:, 0] += 0.25
    fit = fit_rfx(L)
    closed = exceedance_probability(fit["alpha"])
    sampled = exceedance_probability(fit["alpha"] + 0.0, n_samples=400_000, seed=1)
    # force the sampling path by pretending K>2 is not special
    rng2 = np.random.default_rng(1)
    draws = rng2.dirichlet(fit["alpha"], size=400_000)
    mc = np.bincount(draws.argmax(axis=1), minlength=2) / 400_000
    good = abs(closed[0] - mc[0]) < 0.005
    ok &= good
    if verbose:
        print(f"  3 K=2 closed form: {closed[0]:.4f} vs sampled {mc[0]:.4f}  "
              f"{'OK' if good else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    print("validating rfx_bms:")
    print("all checks passed" if validate_rfx_bms() else "VALIDATION FAILED")
