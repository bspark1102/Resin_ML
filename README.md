# ML Surrogate-based Resin Formulation

This repository contains a Jupyter notebook for fitting and evaluating support vector regression (SVR) models that predict the properties of UV-curable resin formulations. The notebook also includes a differential evolution (DE) optimizer for suggesting optimal compositions based on user-defined weights.

---

## Data Layout

All input data are in `.npy` format, located in two folders:

```
train_data/
├── X.npy                # (33, 4) - PUDA, IBOA, EHMA, MAA
├── y_viscosity.npy      # (33,)
├── y_elongation.npy     # (33,)
├── y_modulus.npy        # (33,)
└── y_reactiontime.npy   # (33,)

val_data/
├── X.npy                # (3, 4)
├── y_viscosity.npy
├── y_elongation.npy
├── y_modulus.npy
└── y_reactiontime.npy
```


---

## Notebook Contents

### 1. Training (33 points)

- Loads data from `train_data/*.npy`
- Normalizes `X_tr` to **parts-of-10**:
- Applies log normalization to each target
- Fits four RBF-SVRs with fixed hyperparameters:
  - **Viscosity**: `SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)`
  - **Elongation**: `SVR(kernel='rbf', C=400, gamma=0.01, epsilon=0.01)`
  - **Modulus**: `SVR(kernel='rbf', C=1000, gamma=0.7, epsilon=0.05)`
  - **Reaction Time**: `SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)`

---

### 2. Validation + Visualization + Train metrics

- Loads `val_data/*.npy` and normalizes `X_val`
- Predicts on train and validation splits (with `np.expm1`)
- Prints **metrics**:
  - R² on raw scale
  - log-RMSE (RMSE in log1p space)
- Visualization:
  - Single 2×2 subplot of Actual vs Predicted
  - Panels: Viscosity, Elongation, Modulus, Reaction Time
  - Train points = circles, Validation points = squares
  - y = x reference line included

---

### 3. Differential Evolution optimization (weights are tunable)

- Objective:

<div align="center">

$${\text{score}} = w_E \cdot \text{Elongation} - w_M \cdot \text{Modulus} - w_V \cdot \text{Viscosity} - w_R \cdot \text{ReactionTime}$$

</div>


- Decision variables: parts-of-10 composition `[PUDA, IBOA, EHMA, MAA]`
- Default bounds:
- PUDA: 4.0–5.0
- IBOA: 3.0–5.0
- EHMA: 0.2–5.0
- MAA: 0.4–1.0
- Mass balance enforced via quadratic penalty: `penalty = λ * (sum(x) - 10)^2`
- Uses `scipy.optimize.differential_evolution` with `seed=42`
- Minimal output: only prints optimal composition as fractions

- Tunable parameters:
- Weights: `wE`, `wM`, `wV`, `wR`
- DE parameters: `maxiter`, `popsize`, `seed`

---

## How to Run

1. Ensure `train_data/` and `val_data/` follow the structure above.
2. Open `MLR_formulation.ipynb`
3. Run cells in order:
 - **1** Train models
 - **2** Print train metrics, show 2×2 scatter, (optionally export CSV and sanity check)
 - **3** Run DE optimization (edit weights/bounds as needed)

---

## Reproducibility & Compatibility

- `differential_evolution(..., seed=42)` for deterministic optimization
- Column order of `X.npy` must be exactly: `[PUDA, IBOA, EHMA, MAA]`
- Expected shapes:
- Train: `X (33, 4)`, each `y_* (33,)`
- Val: `X (3, 4)`, each `y_* (3,)`



