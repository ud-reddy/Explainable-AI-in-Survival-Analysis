"""
Explaining Machine Learning Survival Models with Shapley Values
===============================================================
MSc Dissertation, MATH5872M, University of Leeds.

Reproduces every figure and numerical result in the dissertation.
Both datasets load from package loaders -- no local data files needed.
Seed fixed at 42 throughout, so results reproduce exactly.

Usage:  python survshap_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import statsmodels.api as sm
from scipy import stats

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from survshap import SurvivalModelExplainer, PredictSurvSHAP

SEED = 42
PALETTE = ['#d73027', '#2166ac', '#1a9850', '#762a83', '#f46d43',
           '#4575b4', '#74add1', '#fdae61', '#a6d96a']


# ============================================================================
# PART 1 -- SHAP on the diabetes regression benchmark   [Chapter 5.1-5.2]
# ============================================================================

def part1_regression_shap():
    """Fit a Random Forest, verify the efficiency axiom, plot beeswarm and
    waterfall.  Returns objects reused by Parts 2 and 3."""
    print("\n" + "=" * 70)
    print("PART 1 -- SHAP on regression (diabetes)")
    print("=" * 70)

    feat_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    X_raw, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X_raw, columns=feat_names)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               random_state=SEED).fit(X_tr, y_tr)
    print(f"Random Forest R2 = {r2_score(y_te, rf.predict(X_te)):.3f}")

    # TreeSHAP is exact for tree ensembles
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_te.values)
    base = float(explainer.expected_value[0])

    # ---- Efficiency axiom:  sum(phi) == f(x) - E[f(X)] ----
    i = 5
    pred = float(rf.predict(X_te.values[i:i + 1])[0])
    total = base + shap_values[i].sum()
    print(f"\nEfficiency check, patient {i}:")
    print(f"  E[f(X)] = {base:.3f}")
    print(f"  sum phi = {shap_values[i].sum():.3f}")
    print(f"  sum     = {total:.3f}   f(x) = {pred:.3f}")
    print(f"  holds   = {abs(total - pred) < 1e-6}")

    imp = pd.Series(np.abs(shap_values).mean(0), index=feat_names)
    print("\nGlobal importance (mean |phi|):")
    print(imp.sort_values(ascending=False).round(3).to_string())

    # ---- Figure 5.1: beeswarm ----
    plt.figure(figsize=(8, 4.5))
    shap.summary_plot(shap_values, X_te, feature_names=feat_names,
                      show=False, plot_type="dot", max_display=10)
    plt.title("SHAP beeswarm -- diabetes (Random Forest)", fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig("c5_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.close()

    # ---- Figure 5.2: waterfall ----
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(values=shap_values[i], base_values=base,
                         data=X_te.values[i], feature_names=feat_names),
        show=False)
    plt.tight_layout()
    plt.savefig("c5_waterfall.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("\n-> c5_beeswarm.png, c5_waterfall.png")

    return X, X_tr, X_te, y_tr, y_te, rf, shap_values, feat_names


# ============================================================================
# PART 2 -- Linear model comparison under collinearity   [Chapter 5.3]
# ============================================================================

def part2_linear_vs_rf(X_tr, X_te, y_tr, y_te, rf, rf_shap, feat_names):
    """For a linear model, phi_i = beta_i (x_i - xbar_i) exactly.  Comparing
    with RF attributions exposes collinearity-driven misattribution."""
    print("\n" + "=" * 70)
    print("PART 2 -- Linear vs Random Forest (collinearity)")
    print("=" * 70)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    lm = LinearRegression().fit(X_tr_s, y_tr)
    print(f"Linear R2 = {r2_score(y_te, lm.predict(X_te_s)):.3f}")

    lm_shap = shap.LinearExplainer(lm, X_tr_s).shap_values(X_te_s)

    comp = pd.DataFrame({
        "Linear mean|phi|": np.abs(lm_shap).mean(0).round(3),
        "RF mean|phi|":     np.abs(rf_shap).mean(0).round(3),
    }, index=feat_names).sort_values("RF mean|phi|", ascending=False)
    print("\n", comp.to_string())
    print("\nNote s1: the linear model cannot separate s1 from s2-s4, which are")
    print("correlated serum fractions, and over-attributes to it.")

    li, ri = np.abs(lm_shap).mean(0), np.abs(rf_shap).mean(0)
    order = np.argsort(-ri)
    x = np.arange(len(feat_names))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - .2, li[order] / li.max(), .4, label="Linear",
              color="#d73027", alpha=.85)
    ax[0].bar(x + .2, ri[order] / ri.max(), .4, label="Random Forest",
              color="#2166ac", alpha=.85)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([feat_names[k] for k in order], rotation=45, ha="right")
    ax[0].set_ylabel("normalised mean |phi|")
    ax[0].set_title("Feature importance under both models", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].scatter(li, ri, s=80, c="#1a9850", zorder=3, edgecolor="k", lw=.4)
    for k, f in enumerate(feat_names):
        ax[1].annotate(f, (li[k], ri[k]), fontsize=8,
                       xytext=(5, 3), textcoords="offset points")
    lim = max(li.max(), ri.max()) * 1.1
    ax[1].plot([0, lim], [0, lim], "k--", lw=.9, label="exact agreement")
    ax[1].set_xlabel("Linear mean |phi|")
    ax[1].set_ylabel("RF mean |phi|")
    ax[1].set_title("s1 diverges sharply; bmi and s5 agree", fontsize=10)
    ax[1].legend(fontsize=8)

    plt.suptitle("Linear model versus Random Forest SHAP -- diabetes", fontsize=12)
    plt.tight_layout()
    plt.savefig("c5_lmrf.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("\n-> c5_lmrf.png")


# ============================================================================
# PART 3 -- Shapley regression, scalar case   [Chapter 5.4]
# ============================================================================

def part3_shapley_regression(y_te, shap_values, feat_names):
    """Joseph (2019): regress the observed outcome on the SHAP values.
    If the model is calibrated the coefficient should be 1, not merely
    non-zero.  Two tests per feature: beta=0 and beta=1."""
    print("\n" + "=" * 70)
    print("PART 3 -- Shapley regression (Joseph 2019), scalar case")
    print("=" * 70)

    ols = sm.OLS(y_te, sm.add_constant(shap_values)).fit(cov_type="HC3")
    b, se, p0 = ols.params[1:], ols.bse[1:], ols.pvalues[1:]
    # two-sided test of H0: beta = 1
    p1 = np.array([2 * stats.t.sf(abs((b[k] - 1) / se[k]), df=ols.df_resid)
                   for k in range(len(feat_names))])

    print(f"intercept = {ols.params[0]:.2f}   R2 = {ols.rsquared:.3f}")
    print(f"\n{'feature':8s}{'beta':>9}{'SE':>8}{'p(b=0)':>9}{'p(b=1)':>9}")
    print("-" * 43)
    for k, f in enumerate(feat_names):
        print(f"{f:8s}{b[k]:9.3f}{se[k]:8.3f}{p0[k]:9.3f}{p1[k]:9.3f}")

    # Shapley shares: signed relative importance
    ma = np.abs(shap_values).mean(0)
    gamma = np.sign(b) * ma / ma.sum()
    ci = ols.conf_int()
    lo = np.array([ci[k + 1][0] for k in range(len(feat_names))])
    hi = np.array([ci[k + 1][1] for k in range(len(feat_names))])

    order = np.argsort(-np.abs(gamma))
    cols = ["#d73027" if v > 0 else "#4575b4" for v in gamma[order]]
    alphas = [1.0 if p < .05 else .3 for p in p0[order]]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    bars = ax[0].bar(range(len(feat_names)), gamma[order], color=cols)
    for bar, al in zip(bars, alphas):
        bar.set_alpha(al)
    for j, k in enumerate(order):
        star = ("***" if p0[k] < .01 else "**" if p0[k] < .05
                else "*" if p0[k] < .1 else "")
        if star:
            ax[0].text(j, gamma[k] + .004 * np.sign(gamma[k]), star,
                       ha="center", va="bottom" if gamma[k] > 0 else "top",
                       fontsize=9)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xticks(range(len(feat_names)))
    ax[0].set_xticklabels([feat_names[k] for k in order], rotation=45, ha="right")
    ax[0].set_ylabel("Shapley share")
    ax[0].set_title("Shapley shares (faded = not sig. at 5%)", fontsize=10)

    for j, k in enumerate(order):
        ax[1].errorbar(j, b[k], yerr=[[b[k] - lo[k]], [hi[k] - b[k]]],
                       fmt="o", color=cols[j], capsize=4, alpha=alphas[j], ms=6)
    ax[1].axhline(1, color="green", lw=1.5, ls="--", label="beta=1 (calibrated)")
    ax[1].axhline(0, color="grey", lw=.7, ls=":", label="beta=0 (no effect)")
    ax[1].set_xticks(range(len(feat_names)))
    ax[1].set_xticklabels([feat_names[k] for k in order], rotation=45, ha="right")
    ax[1].set_ylabel("beta (95% CI)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Shapley regression coefficients", fontsize=10)

    plt.suptitle("Shapley regression on the diabetes Random Forest", fontsize=12)
    plt.tight_layout()
    plt.savefig("c5_shapreg.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("\n-> c5_shapreg.png")


# ============================================================================
# PART 4 -- Survival models: fitting and evaluation   [Chapter 6.1]
# ============================================================================

def part4_survival_models():
    print("\n" + "=" * 70)
    print("PART 4 -- Survival model fitting (GBSG)")
    print("=" * 70)

    X_raw, y = load_gbsg2()
    Xe = OneHotEncoder().fit_transform(X_raw)
    feats = list(Xe.columns)

    n_ev = y["cens"].sum()
    print(f"n = {len(y)}   events = {n_ev} ({100*n_ev/len(y):.1f}%)   "
          f"censored = {len(y)-n_ev} ({100*(len(y)-n_ev)/len(y):.1f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xe, y, test_size=0.25, random_state=SEED)

    cox = CoxPHSurvivalAnalysis(alpha=0.1).fit(X_tr, y_tr)
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                               min_samples_leaf=5, random_state=SEED,
                               n_jobs=-1).fit(X_tr, y_tr)
    # The GBM answers a narrower question than it might appear to: whether the
    # importance reversal depends on the RSF's particular use of trees.  RSF
    # averages Nelson-Aalen estimates over bootstrapped trees split on the
    # log-rank statistic; GBM fits an additive expansion to the Cox partial
    # likelihood gradient.  Agreement rules out log-rank splitting and bagging
    # as explanations.  It does NOT rule out the bias both share -- tree methods
    # favour continuous covariates over binary ones -- so agreement here is not
    # evidence that the reversal is real.  That comes from Sauerbrei et al.
    # (1999), a fractional polynomial analysis of the same cohort with no
    # splitting mechanism at all.  See Section 6.3.2 of the dissertation.
    gbm = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                           max_depth=3,
                                           random_state=SEED).fit(X_tr, y_tr)

    fns = {"RSF": rsf.predict_survival_function(X_te),
           "GBM": gbm.predict_survival_function(X_te),
           "Cox": cox.predict_survival_function(X_te)}
    ev_times = fns["RSF"][0].x[5:-5]

    print(f"\n{'model':6s}{'C-index':>10}{'IBS':>8}")
    print("-" * 24)
    for name, model in [("RSF", rsf), ("Cox", cox), ("GBM", gbm)]:
        arr = np.row_stack([f(ev_times) for f in fns[name]])
        ibs = integrated_brier_score(y_tr, y_te, arr, ev_times)
        print(f"{name:6s}{model.score(X_te, y_te):10.3f}{ibs:8.3f}")

    # Kaplan-Meier for the full cohort (Figure 4.1)
    t_km, s_km, conf = kaplan_meier_estimator(y["cens"], y["time"],
                                              conf_type="log-log")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.step(t_km / 365, s_km, where="post", color="#2166ac", lw=2,
            label="KM estimate")
    ax.fill_between(t_km / 365, conf[0], conf[1], step="post", alpha=.18,
                    color="#2166ac", label="95% confidence band")
    ax.axhline(.5, color="grey", lw=.8, ls="--", alpha=.7)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Recurrence-free survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("fig_km_gbsg.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("\n-> fig_km_gbsg.png")

    return X_tr, X_te, y_tr, y_te, feats, cox, rsf, gbm, fns


# ============================================================================
# PART 5 -- SurvSHAP(t): curves, ratio diagnostic, importance  [Ch 6.2-6.3]
# ============================================================================

def survshap_one(model, X_tr, y_tr, X_te, idx, times):
    """Kernel SurvSHAP(t) for a single patient."""
    ex = SurvivalModelExplainer(model=model, data=X_tr, y=y_tr)
    ps = PredictSurvSHAP(calculation_method="kernel", random_state=SEED)
    ps.fit(ex, new_observation=X_te.iloc[[idx]], timestamps=times)
    return ps.result


def part5_survshap(X_tr, X_te, y_tr, y_te, feats, cox, rsf, gbm, fns):
    print("\n" + "=" * 70)
    print("PART 5 -- SurvSHAP(t) curves, ratio diagnostic, importance")
    print("=" * 70)

    times = fns["RSF"][0].x[5:-5:20]
    d_rsf = survshap_one(rsf, X_tr, y_tr, X_te, 0, times)
    d_cox = survshap_one(cox, X_tr, y_tr, X_te, 0, times)

    t_cols = [c for c in d_cox.columns if c.startswith("t = ")]
    t_yrs = np.array([float(c.split("= ")[1]) / 365 for c in t_cols])

    # ---- Figure 6.1: attribution curves, stacked for legibility ----
    fig, ax = plt.subplots(2, 1, figsize=(9, 8.4), sharex=True)
    for a, (d, title) in zip(ax, [(d_rsf, "Random Survival Forest"),
                                  (d_cox, "Cox proportional hazards")]):
        for j, (_, row) in enumerate(d.iterrows()):
            a.plot(t_yrs, row[t_cols].values.astype(float),
                   color=PALETTE[j % 9], lw=2.4, label=row["variable_name"])
        a.axhline(0, color="grey", lw=.8, ls="--")
        a.set_ylabel(r"$\phi_i(t)$", fontsize=12)
        a.set_title(title, fontsize=12, pad=6)
        a.grid(alpha=.18)
    ax[1].set_xlabel("Time (years)", fontsize=12)
    ax[0].legend(fontsize=10, ncol=3, loc="upper left", framealpha=.95)
    plt.suptitle("SurvSHAP(t) attribution curves -- test patient 0, GBSG",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("f_curves.png", dpi=190, bbox_inches="tight")
    plt.close()

    # ---- Figure 6.2: proportionality diagnostic (Conjecture 3.1(i)) ----
    # If phi_i(t) = c(t) * beta_i * (x_i - xbar_i) with c(t) shared, then the
    # ratio phi_i(t)/phi_j(t) is constant because c(t) cancels.
    ref = "pnodes"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    cvs = {}
    for a, (d, title, key) in zip(
            ax, [(d_cox, "Cox PH -- ratios stay flat", "Cox"),
                 (d_rsf, "RSF -- ratios drift and cross", "RSF")]):
        den = d[d["variable_name"] == ref][t_cols].values.flatten().astype(float)
        mask = np.abs(den) > 0.004          # drop near-zero denominators
        spreads = []
        for j, (_, row) in enumerate(d.iterrows()):
            if row["variable_name"] == ref:
                continue
            ratio = (row[t_cols].values.astype(float) / den)[mask]
            spreads.append(np.nanstd(ratio) / (abs(np.nanmean(ratio)) + 1e-9))
            a.plot(t_yrs[mask], ratio, "o-", color=PALETTE[j % 9],
                   lw=2, ms=3.5, label=row["variable_name"])
        cvs[key] = np.nanmedian(spreads)
        a.axhline(0, color="grey", lw=.8, ls=":")
        a.set_xlabel("Time (years)", fontsize=11)
        a.set_title(title, fontsize=11.5)
        a.grid(alpha=.18)
        a.set_ylim(-6, 6)
    ax[0].set_ylabel(r"$\phi_i(t)/\phi_{pnodes}(t)$", fontsize=12)
    ax[1].legend(fontsize=8.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.suptitle("Testing Conjecture 3.1(iii): constancy of ratios", fontsize=12.5)
    plt.tight_layout()
    plt.savefig("f_ratio.png", dpi=190, bbox_inches="tight")
    plt.close()

    print(f"\nMedian coefficient of variation of the ratios:")
    print(f"  Cox = {cvs['Cox']:.3f}   RSF = {cvs['RSF']:.3f}   "
          f"(factor {cvs['RSF']/cvs['Cox']:.1f})")

    # ---- Aggregated importance across 5 patients ----
    def mean_importance(model, n=5):
        acc = []
        for i in range(n):
            d = survshap_one(model, X_tr, y_tr, X_te, i, times)
            tc = [c for c in d.columns if c.startswith("t = ")]
            d = d.copy()
            d["imp"] = d[tc].abs().mean(axis=1)
            acc.append(d.set_index("variable_name")["imp"])
        return pd.concat(acc, axis=1).mean(axis=1)

    rsf_imp = mean_importance(rsf)
    gbm_imp = mean_importance(gbm)
    cox_coef = pd.Series(cox.coef_, index=feats).abs()

    tab = pd.DataFrame({"Cox |beta|": cox_coef.round(3),
                        "RSF mean|phi|": rsf_imp.round(4),
                        "GBM mean|phi|": gbm_imp.round(4)})
    print("\nFeature importance -- the reversal:")
    print(tab.sort_values("Cox |beta|", ascending=False).to_string())
    print("\nCox ranks tumour grade top; both ML models rank it last.")
    print("RSF/GBM agreement excludes RSF-specific fitting as the cause,")
    print("but not the tree bias they share -- see Sauerbrei et al. (1999).")

    common = [f for f in feats if f in rsf_imp.index and f in gbm_imp.index]
    norm = lambda s: s[common].values / s[common].values.max()
    cn, rn, gn = norm(cox_coef), norm(rsf_imp), norm(gbm_imp)
    order = np.argsort(-rn)
    x = np.arange(len(common))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax[0].bar(x - .26, cn[order], .26, label="Cox |beta|",
              color="#d73027", alpha=.88)
    ax[0].bar(x, rn[order], .26, label="RSF SurvSHAP(t)",
              color="#2166ac", alpha=.88)
    ax[0].bar(x + .26, gn[order], .26, label="GBM SurvSHAP(t)",
              color="#1a9850", alpha=.88)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([common[k] for k in order], rotation=45,
                          ha="right", fontsize=8)
    ax[0].set_ylabel("normalised importance")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Cox ranks tumour grade first; ML models rank it last",
                    fontsize=10)

    ax[1].scatter(rn, gn, s=80, c="#762a83", zorder=3, edgecolor="k", lw=.4)
    for k, f in enumerate(common):
        ax[1].annotate(f, (rn[k], gn[k]), fontsize=7.5,
                       xytext=(5, 3), textcoords="offset points")
    ax[1].plot([0, 1.08], [0, 1.08], "k--", lw=.9, label="exact agreement")
    ax[1].set_xlabel("RSF importance")
    ax[1].set_ylabel("GBM importance")
    ax[1].set_title("Agreement rules out RSF-specific fitting,\n"
                    "not the shared tree bias", fontsize=9.5)
    ax[1].legend(fontsize=8)

    plt.suptitle("Feature importance: Cox vs RSF vs GBM -- GBSG", fontsize=12)
    plt.tight_layout()
    plt.savefig("f_import.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("\n-> f_curves.png, f_ratio.png, f_import.png")


# ============================================================================
# PART 6 -- Exact SurvSHAP(t) and per-time-point Shapley regression [Ch 6.4]
# ============================================================================

def exact_survshap_factory(cox, X_tr, n_background=80):
    """Exact interventional SurvSHAP(t) by enumerating all 2^p coalitions.

    Feasible here because p = 9 gives only 512 subsets.  Using exact values
    means the efficiency identity holds to machine precision, so no
    approximation error contaminates the regression inference.
    """
    beta = cox.coef_
    X_tr_np = X_tr.values
    p = len(beta)

    # Recover the baseline cumulative hazard H0 from any fitted prediction:
    # H(t|x) = H0(t) * exp(beta'x)  =>  H0(t) = H(t|x0) / exp(beta'x0)
    fn0 = cox.predict_cumulative_hazard_function(X_tr.iloc[[0]])[0]
    eta0 = float(X_tr_np[0] @ beta)
    H0 = lambda t: float(fn0(t)) / np.exp(eta0)
    S_of = lambda M, h: np.exp(-h * np.exp(M @ beta))

    rng = np.random.RandomState(0)
    BG = X_tr_np[rng.choice(len(X_tr_np), n_background, replace=False)]

    all_subsets = [S for r in range(p + 1) for S in combinations(range(p), r)]
    weights = {
        k: [(tuple(sorted(S + (k,))), tuple(sorted(S)),
             factorial(len(S)) * factorial(p - len(S) - 1) / factorial(p))
            for r in range(p)
            for S in combinations([j for j in range(p) if j != k], r)]
        for k in range(p)
    }

    def phi(x, t):
        h = H0(t)
        v = {}
        for S in all_subsets:
            M = BG.copy()
            if S:
                M[:, list(S)] = x[list(S)]
            v[S] = S_of(M, h).mean()
        return np.array([sum(w * (v[a] - v[b]) for a, b, w in weights[k])
                         for k in range(p)])

    return phi


def part6_shapley_regression_over_time(X_tr, X_te, y_tr, y_te, feats, cox):
    """Regress the OBSERVED survival indicator on phi_k(t) at each time point.

    Two facts give beta = 1 in population:
      (1) efficiency:   sum_k phi_k(t|x) = S(t|x) - E[S(t|X)]
      (2) the bridge:   E[1{T>t} | x]    = P(T>t|x) = S(t|x)
    Neither uses the Cox linear predictor, so the result is model-agnostic.

    NOTE: regressing the model's own S(t|x) on its own phi_k(t) would be an
    algebraic identity returning beta == 1 for any model whatsoever.  The
    response must be the observed outcome for beta = 1 to be falsifiable.
    """
    print("\n" + "=" * 70)
    print("PART 6 -- Shapley regression at each time point")
    print("=" * 70)

    phi_fn = exact_survshap_factory(cox, X_tr)
    X_te_np = X_te.values
    T = np.asarray(y_te["time"])
    E = np.asarray(y_te["cens"])
    N, p = len(T), len(feats)

    t_grid = np.percentile(T, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    B = np.zeros((len(t_grid), p))
    SE = np.zeros((len(t_grid), p))
    for m, t in enumerate(t_grid):
        PHI = np.array([phi_fn(X_te_np[i], t) for i in range(N)])
        # censoring is ignored here: the response uses observed time directly
        ols = sm.OLS((T > t).astype(float),
                     sm.add_constant(PHI)).fit(cov_type="HC3")
        B[m], SE[m] = ols.params[1:], ols.bse[1:]
        print(f"  t = {t:6.0f} d   fitted")

    # censoring survival function G(t): flip the event indicator in KM
    tg, Gs = kaplan_meier_estimator(~E, T)
    G = lambda t: Gs[max(np.searchsorted(tg, t, side="right") - 1, 0)]
    Gt = np.array([G(t) for t in t_grid])

    mean_se = SE.mean(0)
    dev = np.abs(B - 1).mean(0)
    order = np.argsort(mean_se)
    well, poorly = order[:4], order[4:]

    summary = pd.DataFrame({
        "Cox beta": pd.Series(cox.coef_, index=feats).round(3),
        "mean SE": pd.Series(mean_se, index=feats).round(3),
        "mean beta_k": pd.Series(B.mean(0), index=feats).round(3),
    }).sort_values("mean SE")
    print("\n", summary.to_string())
    print(f"\ncorr(log SE, |beta-1|) = "
          f"{np.corrcoef(np.log(mean_se), dev)[0,1]:.3f}")
    print("Well-identified coefficients sit near 1; poorly-identified ones wander.")

    t_yrs = t_grid / 365

    # ---- Figure 6.3: beta_k(t), split by identification ----
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for k in well:
        ax[0].plot(t_yrs, B[:, k], "o-", lw=1.9, ms=4.5, label=feats[k])
        ax[0].fill_between(t_yrs, B[:, k] - 1.96 * SE[:, k],
                           B[:, k] + 1.96 * SE[:, k], alpha=.09)
    for k in poorly:
        ax[1].plot(t_yrs, B[:, k], "o-", lw=1.5, ms=4, alpha=.85, label=feats[k])
    for a, title, colour in [(ax[0], "Well identified (low SE)", "#1a5fb4"),
                             (ax[1], "Poorly identified (high SE)", "#c0392b")]:
        a.axhline(1, color="green", lw=1.7, ls="--", label="beta=1")
        a.axhline(0, color="grey", lw=.7, ls=":")
        a.set_xlabel("Time (years)")
        a.set_title(title, fontsize=10.5, color=colour)
        a.legend(fontsize=7.5, loc="upper left")
    ax[0].set_ylabel(r"$\hat\beta_k(t)$")
    ax[0].set_ylim(-8, 9)
    plt.suptitle("Shapley regression coefficient over time -- Cox model",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("f_shapreg.png", dpi=170, bbox_inches="tight")
    plt.close()

    # ---- Figure 6.4: censoring attenuates beta to G(t) ----
    # Ignoring censoring, E[1{min(T,C)>t}|x] = S(t|x) G(t), so beta = G(t).
    mb = B[:, well].mean(1)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(t_yrs, Gt, "s-", color="#8e44ad", lw=2, ms=6, label="G(t)")
    ax.plot(t_yrs, mb, "o-", color="#1a5fb4", lw=2, ms=6,
            label="mean beta_k(t)")
    ax.axhline(1, color="green", lw=1.5, ls="--", label="beta=1")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.8)
    ax.set_title("Ignoring censoring attenuates beta to G(t)", fontsize=11)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig("f_cens.png", dpi=170, bbox_inches="tight")
    plt.close()

    print(f"\nCensoring attenuation, last time point: "
          f"G = {Gt[-1]:.3f}, mean beta = {mb[-1]:.3f}")
    print("\n-> f_shapreg.png, f_cens.png")


# ============================================================================
def main():
    X, X_tr, X_te, y_tr, y_te, rf, rf_shap, feat_names = part1_regression_shap()
    part2_linear_vs_rf(X_tr, X_te, y_tr, y_te, rf, rf_shap, feat_names)
    part3_shapley_regression(y_te, rf_shap, feat_names)

    (sX_tr, sX_te, sy_tr, sy_te, feats,
     cox, rsf, gbm, fns) = part4_survival_models()
    part5_survshap(sX_tr, sX_te, sy_tr, sy_te, feats, cox, rsf, gbm, fns)
    part6_shapley_regression_over_time(sX_tr, sX_te, sy_tr, sy_te, feats, cox)

    print("\n" + "=" * 70)
    print("Done. All figures written to the working directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
