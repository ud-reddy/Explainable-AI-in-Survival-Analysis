# Explaining Machine Learning Survival Models with Shapley Values

Code accompanying an MSc dissertation submitted for MATH5872M, School of
Mathematics, University of Leeds.

The project applies the SurvSHAP(t) framework of Krzyzinski et al. (2023) to
machine learning survival models, and extends the Shapley regression of
Joseph (2019) to the survival setting by fitting it separately at each time point.

---

## Quick start

```bash
pip install -r requirements.txt
python survshap_analysis.py
```

Or open `shapley_analysis.ipynb`, which contains the same analysis with all
outputs and figures already embedded — no need to run anything to see the results.

Neither dataset needs downloading. Both load from package loaders
(`sklearn.datasets.load_diabetes`, `sksurv.datasets.load_gbsg2`). The random seed
is fixed at 42 throughout, so every reported figure reproduces exactly.

Runtime is about 4 minutes on a modern laptop. The bulk of it is Part 6, which
computes *exact* SurvSHAP(t) values by enumerating all 2⁹ = 512 coalitions for
every patient at every time point.

---

## Contents

| File | Description |
|------|-------------|
| `survshap_analysis.py` | Main script. Loads both datasets, fits all models, computes SHAP and SurvSHAP(t) values, runs the Shapley regressions, writes every figure. |
| `shapley_analysis.ipynb` | Notebook version with outputs and figures embedded. |
| `requirements.txt` | Pinned package versions, matching Table 4.2 of the dissertation. |

## Structure of the analysis

| Part | Dissertation section | Figures produced |
|------|---------------------|------------------|
| 1 | 5.1–5.2 — SHAP on the diabetes benchmark; efficiency axiom check | `c5_beeswarm.png`, `c5_waterfall.png` |
| 2 | 5.3 — linear model vs Random Forest under collinearity | `c5_lmrf.png` |
| 3 | 5.4 — Shapley regression, scalar case | `c5_shapreg.png` |
| 4 | 6.1 — survival model fitting and evaluation | `fig_km_gbsg.png` |
| 5 | 6.2–6.3 — SurvSHAP(t) curves, proportionality diagnostic, importance | `f_curves.png`, `f_ratio.png`, `f_import.png` |
| 6 | 6.4 — Shapley regression at each time point | `f_shapreg.png`, `f_cens.png` |

---

## Main results

### Predictive performance (GBSG, n = 686, 43.6% events)

| Model | C-index | IBS |
|-------|---------|-----|
| Random Survival Forest | **0.667** | **0.171** |
| Cox proportional hazards | 0.644 | 0.175 |
| Gradient boosting | 0.641 | 0.177 |

All three are comparable, so divergence in their explanations reflects a
difference in learned structure rather than in fit quality.

### The feature importance reversal

Cox ranks tumour grade first and second; both machine learning models rank
progesterone receptor level and lymph node count first and second, with tumour
grade last. Spearman rank correlation between the Cox and RSF orderings is −0.62.

| Feature | Cox \|β\| | RSF mean\|φ\| | GBM mean\|φ\| |
|---------|-----------|---------------|---------------|
| `tgrade=III` | 0.775 | 0.0034 | 0.0014 |
| `tgrade=II` | 0.619 | 0.0029 | 0.0000 |
| `pnodes` | 0.048 | 0.0650 | **0.0757** |
| `progrec` | 0.003 | **0.0691** | 0.0756 |

The gradient boosting model is included to test whether this depends on the RSF's
*particular* use of trees — it does not. But RSF and GBM are both tree ensembles
and share the same preference for continuous covariates over binary ones, so
their agreement is **not** evidence that the reversal is real. That evidence is
external: Sauerbrei et al. (1999) analysed the identical 686-patient cohort using
fractional polynomials, a parametric method with no splitting mechanism, and also
found `progrec` to require a non-linear transformation.

### The proportionality conjecture

Under a Cox model a first-order expansion gives φ_i(t) ≈ c(t)·β_i·(x_i − x̄_i),
where c(t) is shared across features. The ratio φ_i(t)/φ_j(t) should therefore be
constant in t, since c(t) cancels. Measured median coefficient of variation of
these ratios:

**0.065 for Cox against 0.416 for the RSF — a factor of 6.4.**

Crossing curves are a sufficient condition for non-proportional hazards.

### Shapley regression at each time point

Regressing the observed survival indicator on φ_k(t) gives β_k(t) = 1 in
population, from two facts:

1. **efficiency** — `sum_k φ_k(t|x) = S(t|x) − E[S(t|X)]`
2. **the bridge** — `E[1{T>t} | x] = P(T>t|x) = S(t|x)`

Neither uses the Cox linear predictor, so the result is model-agnostic and should
hold for models with no linear predictor at all.

Empirically, coefficients settle near 1 only where the attributions are well
identified: `pnodes` reaches 1.001 and `tgrade=II` reaches 0.984, while features
carrying little attribution wander. Correlation between log(SE) and |β̂ − 1| is
**0.590**.

### Censoring attenuation

Ignoring censoring, `E[1{min(T,C)>t} | x] = S(t|x)·G(t)`, so the regression
estimates **β_k(t) = G(t)**, not 1, where G is the censoring survival function.
Early in follow-up G ≈ 1 and the attenuation is invisible; later it decays in
step. At the final time point G = 0.207 and the mean coefficient is 0.207.

Inverse probability of censoring weighting by 1/Ĝ(t) should restore β = 1.

### A misspecification diagnostic

`progrec` has the smallest standard error of any feature (0.576) yet a coefficient
furthest from 1 (−0.153). Being well identified, this cannot be noise, so the
calibration assumption must fail for that covariate. This flags the Cox
misspecification using only the Cox fit, with no reference to any ML model:

> small SE **and** β̂ far from 1 ⟹ that covariate is misspecified

---

## One implementation detail worth noting

Part 6 regresses on the **observed** outcome `1{T_i > t}`, not on the model's own
`S(t|x)`. Regressing a model's prediction on its own SHAP values is an algebraic
identity: by efficiency it returns β ≡ 1 with zero residual variance for *any*
model, including a deliberately poor one. Using the observed outcome makes β = 1
a falsifiable hypothesis rather than an arithmetic consequence.

---

## Data

- **Diabetes** — Efron et al. (2004). 442 patients, 10 baseline clinical features,
  continuous outcome measuring disease progression after one year.
- **GBSG** — Schumacher et al. (1994). 686 node-positive breast cancer patients,
  9 features after one-hot encoding. 299 events (43.6%), 387 right-censored.
  Median follow-up 1,084 days.

## References

Ducrot, L. et al. (2025). *SurvTreeSHAP(t): scalable explanation method for
tree-based survival models.* IJCAI XAI Workshop. HAL: hal-05108033.

Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). *Least angle
regression.* Annals of Statistics, 32(2), 407–499.

Ishwaran, H. et al. (2008). *Random survival forests.* Annals of Applied
Statistics, 2(3), 841–860.

Joseph, A. (2019). *Shapley regressions: a framework for statistical inference on
machine learning models.* Bank of England Staff Working Paper No. 784.

Krzyzinski, M., Spytek, M., Baniecki, H., Biecek, P. (2023). *SurvSHAP(t):
time-dependent explanations of machine learning survival models.*
Knowledge-Based Systems, 262, 110234.

Langbein, S.H. et al. (2026). *Functional decomposition and Shapley interactions
for interpreting survival models.* arXiv:2602.16505.

Lundberg, S.M., Lee, S.I. (2017). *A unified approach to interpreting model
predictions.* NeurIPS 30, 4765–4774.

Sauerbrei, W., Royston, P., Bojar, H., Schmoor, C., Schumacher, M. (1999).
*Modelling the effects of standard prognostic factors in node-positive breast
cancer.* British Journal of Cancer, 79(11–12), 1752–1760.

Schumacher, M. et al. (1994). *Randomized 2×2 trial evaluating hormonal treatment
and the duration of chemotherapy in node-positive breast cancer patients.*
Journal of Clinical Oncology, 12(10), 2086–2093.
