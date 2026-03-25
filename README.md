# ML Surrogate-based Resin Formulation

This repository contains a Jupyter notebook for fitting and evaluating support vector regression (SVR) models that predict the properties of UV-curable resin formulations. The notebook also includes a differential evolution (DE) optimizer for suggesting optimal compositions based on user-defined weights.

---

## Data Layout

All input data are in `.npy` format, located in three folders:

```
train_data/
├── X.npy                # (33, 4) - PUDA, IBOA, EHMA, MAA
├── y_viscosity.npy      # (33,)
├── y_elongation.npy     # (33,)
├── y_modulus.npy        # (33,)
└── y_reactiontime2.npy   # (33,)

val_data/
├── X.npy                 # (3, 4)
├── y_viscosity.npy       # (3,)
├── y_elongation.npy      # (3,)
├── y_modulus.npy         # (3,)
└── y_reactiontime2.npy   # (3,)

ext_val_data/
├── X.npy                 # (9, 4)
├── y_viscosity.npy       # (9,)
├── y_elongation.npy      # (9,)
├── y_modulus.npy         # (9,)
└── y_reactiontime2.npy   # (9,)
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
  - **Modulus**: `SVR(kernel='rbf', C=17, gamma=0.04, epsilon=0.13)`
  - **Reaction Time**: `SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)`

---

### 2. Differential Evolution optimization (weights are tunable)

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

- Tunable parameters:
  - Weights: `wE`, `wM`, `wV`, `wR`
  - DE parameters: `maxiter`, `popsize`, `seed`

---

### 3. Validation of Surrogate Models

(5-fold CV + external validation)
This section validates the surrogate in two ways:
1. 5-fold cross-validation on the combined 36-point set (`train+val`)
2. External validation on the 9-point external dataset

### 3.1 5-fold cross-validation on all 36 formulations

The notebook loads `val_data/`, combines it with `train_data/`, and performs 5-fold cross-validation on the full 36-point in-domain dataset.

#### Combined dataset
- Training set: 33 points
- Validation set: 3 points
- Combined CV dataset: 36 points

#### Procedure
- Loads `X_val` and the four targets from `val_data/`
- Converts `X_val` to parts-of-10 if needed
- Stacks `train_data` and `val_data`
- Applies `log1p` normalization to each target
- Runs 5-fold CV with:
  - `KFold(n_splits=5, shuffle=True, random_state=42)`

For each fold, the notebook:
- retrains the four SVR models on the fold training split
- predicts on the fold test split
- inverse-transforms predictions using `np.expm1`
- clips predictions at zero using `np.maximum(..., 0.0)`

#### Reported metrics
For each property, the following are computed:
- `R2_log`
- `RMSE_log`
- `R2_raw`
- `RMSE_raw`

#### Outputs
The notebook prints:
- fold-wise metrics for each fold
- **CV summary (mean ± s.d. across folds)**
- **OOF metrics on all 36 points**

These are reported for:
- Viscosity
- Elongation
- Modulus
- Reaction Time

### 3.2 External validation

The notebook then evaluates the fitted surrogate models on the external validation set stored in `ext_val_data/`.

#### External dataset
- External validation set: 9 points

#### Procedure
- Loads `X_ext` and the four targets from `ext_val_data/`
- Converts `X_ext` to parts-of-10 if needed
- Predicts on:
  - the training set
  - the 3-point `val_data` set
  - the 9-point external validation set

#### Reported metrics
The notebook prints:
- **Train metrics** using the 33 training points
- **All validation metrics** on the combined 12-point validation set:
  - original validation set: 3 points
  - external validation set: 9 points

For each property, the reported metrics are:
- raw-scale `R²`
- log-space `RMSE`

### 3.3 External validation table and parity plot

The notebook also:
- builds a table of external validation predictions
- optionally saves it as:
  - `ext_val_data/external_validation_predictions.csv`
  - `ext_val_data/external_validation_predictions.xlsx`

A 2×2 parity plot is generated for:
- Viscosity
- Elongation
- Young's Modulus
- Reaction Time

Marker styles:
- **Train** = circles
- **Val** = orange squares
- **External Val** = crimson diamonds

Each panel includes a dashed `y = x` reference line.

---

## How to Run

1. Ensure `train_data/`, `val_data/`, and `ext_val_data/` follow the structure above.
2. Open `MLR_formulation.ipynb`
3. Run cells in order:
   - **1.** Fit the final SVR models on the 33-point training set
   - **2.** Run DE optimization using the trained surrogate models
   - **3.** Run validation of surrogate models:
     - 5-fold CV on the combined 36-point `train + val` dataset
     - external validation on the 9-point `ext_val_data` set
     - combined validation metrics and parity plots

The notebook is intended to be executed sequentially. Later sections assume the models from the training section have already been fitted.

---

## Reproducibility & Compatibility

- The column order of all input arrays must be:
  - `[PUDA, IBOA, EHMA, MAA]`

- Reaction time files are expected to use:
  - `y_reactiontime2.npy`

- Input arrays may be stored either:
  - in wt% summing to `100`, or
  - in parts-of-10 summing to `10`

  The notebook converts wt% inputs internally to parts-of-10.

- Target variables are modeled in log-space using:
  - `np.log1p(...)` during training
  - `np.expm1(...)` for inverse transformation during prediction

- 5-fold CV uses:
  - `KFold(n_splits=5, shuffle=True, random_state=42)`

- Differential evolution uses:
  - `seed = 42`

### Expected dataset sizes
- **Train**: `X (33, 4)` and each `y_* (33,)`
- **Val**: `X (3, 4)` and each `y_* (3,)`
- **External Val**: `X (9, 4)` and each `y_* (9,)`

### Evaluation structure
- Final surrogate models are trained on the 33-point training set
- 5-fold CV is performed on the combined 36-point in-domain dataset (`train + val`)
- External validation is evaluated separately using `ext_val_data/`

---

## Notes

- The notebook assumes all target arrays are nonnegative.
- External validation predictions can be exported from the validation section.
- The notebook should be run in order without skipping earlier sections.

