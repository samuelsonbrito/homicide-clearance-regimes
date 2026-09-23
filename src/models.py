"""
Censored maximum-likelihood fits of the three nested clearance models.

  Model 1: F(t) = 1 - exp(-k t)
  Model 2: F(t) = L (1 - exp(-k t))                              (mixture cure)
  Model 3: F(t) = L1 (1 - exp(-k1 t)) + L2 (1 - exp(-k2 t))      (two regimes)

t is measured in months. Rates are optimized on the log scale and shares on
the logit scale so that all constraints hold automatically.
"""
import numpy as np
from scipy.optimize import minimize


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- log-likelihoods ----------
def ll_model2(k, L, T, E):
    return np.where(E, np.log(L * k) - k * T, np.log(1 - L * (1 - np.exp(-k * T)))).sum()


def ll_model3(k1, k2, a, b, T, E):
    L1 = sig(a)
    L2 = sig(b) * (1 - L1)
    f = L1 * k1 * np.exp(-k1 * T) + L2 * k2 * np.exp(-k2 * T)
    F = L1 * (1 - np.exp(-k1 * T)) + L2 * (1 - np.exp(-k2 * T))
    return np.where(E, np.log(f), np.log(1 - F)).sum()


# ---------- fits ----------
def fit_model1(T, E):
    r = minimize(lambda p: -ll_model2(np.exp(p[0]), 1.0, T, E), [np.log(0.3)])
    return {"k": float(np.exp(r.x[0])), "nll": float(r.fun), "aic": float(2 * r.fun + 2)}


def fit_model2(T, E, x0=(0.0, 0.0)):
    r = minimize(lambda p: -ll_model2(np.exp(p[0]), sig(p[1]), T, E), list(x0))
    return {"k": float(np.exp(r.x[0])), "L": float(sig(r.x[1])),
            "nll": float(r.fun), "aic": float(2 * r.fun + 4), "x": r.x}


NM = dict(method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
STARTS = [[3, -1, 0, -1], [4, 0, -0.5, -1], [2, -2, 0, 0], [3.2, -0.9, -1, -1.5]]


def _unpack3(x):
    k1, k2 = np.exp(x[0]), np.exp(x[1])
    L1 = sig(x[2])
    L2 = sig(x[3]) * (1 - L1)
    if k1 < k2:  # label the fast regime as regime 1
        k1, k2, L1, L2 = k2, k1, L2, L1
    return float(k1), float(L1), float(k2), float(L2)


def fit_model3(T, E, starts=STARTS):
    best = None
    for s in starts:
        r = minimize(lambda p: -ll_model3(np.exp(p[0]), np.exp(p[1]), p[2], p[3], T, E), s, **NM)
        if best is None or r.fun < best.fun:
            best = r
    k1, L1, k2, L2 = _unpack3(best.x)
    return {"k1": k1, "L1": L1, "k2": k2, "L2": L2, "L": L1 + L2,
            "half_life_fast_days": float(np.log(2) / k1 * 30.44),
            "half_life_slow_months": float(np.log(2) / k2),
            "nll": float(best.fun), "aic": float(2 * best.fun + 8), "x": best.x}


# ---------- fitted curves ----------
def F1(t, p):
    return 1 - np.exp(-p["k"] * t)


def F2(t, p):
    return p["L"] * (1 - np.exp(-p["k"] * t))


def F3(t, p):
    return p["L1"] * (1 - np.exp(-p["k1"] * t)) + p["L2"] * (1 - np.exp(-p["k2"] * t))


# ---------- Kaplan-Meier ----------
def kaplan_meier(T, E, points):
    o = np.argsort(T)
    T, E = T[o], E[o]
    S, steps = 1.0, []
    for u in np.unique(T[E]):
        S *= 1 - ((T == u) & E).sum() / (T >= u).sum()
        steps.append((u, 1 - S))
    out = []
    for q in points:
        v = [s for u, s in steps if u <= q]
        out.append(v[-1] if v else 0.0)
    return np.array(out)


# ---------- bootstrap ----------
def bootstrap_model2(T, E, x0, n=200, seed=1):
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        i = rng.integers(0, len(T), len(T))
        p = fit_model2(T[i], E[i], x0)
        res.append([p["k"], p["L"]])
    return np.array(res)


def bootstrap_model3(T, E, x0, n=100, seed=1):
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        i = rng.integers(0, len(T), len(T))
        r = minimize(lambda p: -ll_model3(np.exp(p[0]), np.exp(p[1]), p[2], p[3], T[i], E[i]),
                     x0, method="Nelder-Mead", options={"maxiter": 8000})
        res.append(_unpack3(r.x))
    return np.array(res)  # columns: k1, L1, k2, L2


# ---------- pooled cross-state tests ----------
def pooled_tests(data):
    """data: dict state -> (T, E). Returns fits and likelihood ratio tests."""
    from scipy.stats import chi2
    S = list(data)
    opt = dict(method="Nelder-Mead",
               options={"maxiter": 100000, "maxfev": 100000, "xatol": 1e-8, "fatol": 1e-8})

    def refit(f, x0):
        r = minimize(f, x0, **opt)
        for _ in range(3):
            r = minimize(f, r.x, **opt)
        return r

    def free(p):
        return -sum(ll_model3(np.exp(p[4*i]), np.exp(p[4*i+1]), p[4*i+2], p[4*i+3], *data[s])
                    for i, s in enumerate(S))

    def common_slow(p):
        return -sum(ll_model3(np.exp(p[1+3*i]), np.exp(p[0]), p[2+3*i], p[3+3*i], *data[s])
                    for i, s in enumerate(S))

    def common_both(p):
        return -sum(ll_model3(np.exp(p[0]), np.exp(p[1]), p[2+2*i], p[3+2*i], *data[s])
                    for i, s in enumerate(S))

    n = len(S)
    r_free = refit(free, [np.log(24), np.log(0.4), -1, -1] * n)
    r_slow = refit(common_slow, [np.log(0.4)] + [np.log(24), -1, -1] * n)
    r_both = refit(common_both, [np.log(24), np.log(0.4)] + [-1, -1] * n)

    k_free, k_slow, k_both = 4 * n, 1 + 3 * n, 2 + 2 * n
    lr_slow = 2 * (r_slow.fun - r_free.fun)
    lr_both = 2 * (r_both.fun - r_free.fun)
    return {
        "aic_free": 2 * r_free.fun + 2 * k_free,
        "aic_common_slow": 2 * r_slow.fun + 2 * k_slow,
        "aic_common_both": 2 * r_both.fun + 2 * k_both,
        "lr_common_slow": lr_slow, "p_common_slow": chi2.sf(lr_slow, k_free - k_slow),
        "lr_common_both": lr_both, "p_common_both": chi2.sf(lr_both, k_free - k_both),
        "common_k2": float(np.exp(r_slow.x[0])),
        "common_k1_k2": (float(np.exp(r_both.x[0])), float(np.exp(r_both.x[1]))),
    }
