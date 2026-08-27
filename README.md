# Explaining Machine Learning Survival Models with Shapley Values

Code accompanying an MSc dissertation (MATH5872M, University of Leeds).

The project applies the SurvSHAP(t) framework of Krzyzinski et al. (2023) to
machine learning survival models, and extends the Shapley regression of
Joseph (2019) to the survival setting.

## Quick start

```bash
pip install -r requirements.txt
python survshap_analysis.py
```

Neither dataset needs downloading — both load from package loaders
(`sklearn.datasets.load_diabetes`, `sksurv.datasets.load_gbsg2`). The seed is
fixed at 42 throughout, so every reported figure reproduces exactly.

Runtime is roughly 30–40 minutes, dominated by Part 6, which computes exact
SurvSHAP(t) values by enumerating all 2^9 = 512 coalitions for every patient at
every time point.

## What the script does

| Part | Section in dissertation | Output |
|------|------------------------|--------|
| 1 | 5.1–5.2 — SHAP on the diabetes benchmark; efficiency axiom check | `c5_beeswarm.png`, `c5_waterfall.png` |
| 2 | 5.3 — linear model vs Random Forest under collinearity | `c5_lmrf.png` |
| 3 | 5.4 — Shapley regression, scalar case | `c5_shapreg.png` |
| 4 | 6.1 — survival model fitting and evaluation | `fig_km_gbsg.png` |
| 5 | 6.2–6.3 — SurvSHAP(t) curves, proportionality diagnostic, importance | `f_curves.png`, `f_ratio.png`, `f_import.png` |
| 6 | 6.4 — per-time-point Shapley regression | `f_shapreg.png`, `f_cens.png` |

## Main results

**The feature importance reversal (Section 6.3).** Cox ranks tumour grade first
and second; both machine learning models rank progesterone receptor level and
lymph node count first and second, with tumour grade last. Spearman rank
correlation between the Cox and RSF orderings is −0.62.

The GBM is included to test whether this depends on the RSF's particular use of
trees — it does not. But RSF and GBM are both tree ensembles and share the same
preference for continuous covariates over binary ones, so their agreement is
*not* evidence that the reversal is real. That evidence is external: Sauerbrei
et al. (1999) analysed the identical 686-patient cohort using fractional
polynomials, a parametric method with no splitting mechanism, and also found
`progrec` to require a non-linear transformation.

**The proportionality conjecture (Sections 3.5, 6.2).** Under a Cox model,
φ_i(t) ≈ c(t)·β_i·(x_i − x̄_i) with c(t) shared across features, so the ratio
φ_i(t)/φ_j(t) should be constant in t. Measured median coefficient of variation
of these ratios: **0.065 for Cox, 0.416 for RSF** — a factor of 6.4.

**Shapley regression at each time point (Section 6.4).** Regressing the observed
survival indicator on φ_k(t) gives β_k(t) = 1 in population, from two facts:

- efficiency — `sum_k φ_k(t|x) = S(t|x) − E[S(t|X)]`
- the bridge — `E[1{T>t} | x] = P(T>t|x) = S(t|x)`

Neither uses the Cox linear predictor, so the result is model-agnostic.

**Censoring attenuation.** Ignoring censoring gives β_k(t) = G(t), not 1, where
G is the censoring survival function. At the final time point G = 0.207 and the
mean coefficient is 0.207.

**A misspecification diagnostic.** `progrec` has the smallest standard error of
any feature yet a coefficient furthest from 1 — well identified, so not noise.
This flags the Cox misspecification using only the Cox fit, with no reference to
any machine learning model.

## A note on one implementation detail

Part 6 regresses on the **observed** outcome `1{T_i > t}`, not on the model's own
`S(t|x)`. Regressing a model's prediction on its own SHAP values is an algebraic
identity: by efficiency it returns β ≡ 1 with zero residual variance for any
model, including a deliberately poor one. Using the observed outcome makes
β = 1 a falsifiable hypothesis rather than an arithmetic consequence.

## Data

- **Diabetes** — Efron et al. (2004). 442 patients, 10 features, continuous outcome.
- **GBSG** — Schumacher et al. (1994). 686 breast cancer patients, 9 features
  after one-hot encoding; 299 events (43.6%), 387 right-censored.

## References

Joseph, A. (2019). *Shapley regressions: a framework for statistical inference on
machine learning models.* Bank of England Staff Working Paper No. 784.

Krzyzinski, M., Spytek, M., Baniecki, H., Biecek, P. (2023). *SurvSHAP(t):
time-dependent explanations of machine learning survival models.*
Knowledge-Based Systems, 262, 110234.

Lundberg, S.M., Lee, S.I. (2017). *A unified approach to interpreting model
predictions.* NeurIPS 30, 4765–4774.

Ishwaran, H. et al. (2008). *Random survival forests.* Annals of Applied
Statistics, 2(3), 841–860.

Sauerbrei, W. et al. (1999). *Modelling the effects of standard prognostic
factors in node-positive breast cancer.* British Journal of Cancer, 79(11–12),
1752–1760.
