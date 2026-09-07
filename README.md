# Credit Scorecard

A loan-default classifier built from scratch — logistic regression implemented directly in NumPy, with no scikit-learn, trained on the Kaggle [credit risk dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

Run it, wait for training, then type in a loan application and get a verdict: **likely to default** or **safe to grant**.

## Why

Most credit-scoring examples call `LogisticRegression().fit()` and stop there. This one spells out every piece — the sigmoid, the cross-entropy cost, the gradient, the descent loop — so the model is readable end to end rather than a black box behind one import.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib kagglehub
python main.py
```

The dataset downloads automatically on first run via `kagglehub` (and is cached locally afterwards). Kaggle credentials may be required — see [kagglehub authentication](https://github.com/Kaggle/kagglehub#authenticate).

A session looks like this:

```
Training model...

Cost after iteration 0: 0.6931471805599453
Cost after iteration 100: 0.4127...
...
training accuracy: 82.1.
test accuracy: 81.7.

Model trained. Enter a loan application to get a prediction (Ctrl+C to quit).

Applicant age: 27
Annual income: 54000
Home ownership (MORTGAGE/OTHER/OWN/RENT): RENT
...
Prediction: this loan is likely to be repaid - safe to grant.
```

## How it works

```
fetch_dataset()  →  extract_data()  →  to_feature_vector()  →  LogisticRegression
   Kaggle CSV        drop NaN rows      standardize + one-hot     train / predict
```

1. **Fetch** — `src/fetch.py` pulls `credit_risk_dataset.csv` into a pandas `DataFrame`.
2. **Extract** — `src/extraction.py` drops rows missing `person_emp_length` or `loan_int_rate`, leaving zero NaNs.
3. **Encode** — `main.py:to_feature_vector` builds a **25-dimensional** vector per applicant:

   | Range | Fields | Encoding |
   | --- | --- | --- |
   | 0–6 | age, income, employment length, loan amount, interest rate, loan/income ratio, credit history length | standardized (`(x - mean) / std`) |
   | 7–10 | `person_home_ownership` | one-hot |
   | 11–16 | `loan_intent` | one-hot |
   | 17–23 | `loan_grade` | one-hot |
   | 24 | `cb_person_default_on_file` | binary flag |

   Standardization matters: raw incomes run into the tens of thousands and would overflow the sigmoid during training.
4. **Train** — an 80/20 split (seeded at 42), then 2000 iterations of batch gradient descent at a learning rate of 0.1. Accuracy on both splits is printed at the end.
5. **Predict** — new applications are parsed at the prompt and pushed through the *same* encoding path, reusing the training set's mean and std so the scales line up.

## Layout

```
main.py                    Training + interactive prediction loop
src/
  fetch.py                 Downloads and loads the Kaggle dataset
  extraction.py            Cleaning: drops rows with missing values
  primitives.py            Categorical enums and one-hot encoding
  parse.py                 Prompts for a single loan application
  model.py                 LogisticRegression, implemented in NumPy
dataset.ipynb              Dataset preview: columns, dtypes, sample rows
data_extraction.ipynb      Cleaning walkthrough + NaN counts before/after
data_analysis.ipynb        Summary statistics and per-column distributions
data_preprocessing.ipynb   Feature-vector construction, step by step
```

The four notebooks mirror the pipeline stages and are the place to look if you want to *see* the data rather than just run the model.

## The model

`src/model.py` is a self-contained `LogisticRegression`:

| Method | Role |
| --- | --- |
| `sigmoid` | squashes a real score into a (0, 1) probability |
| `hypothesis` | `sigmoid(wᵀX + b)` — current predicted probabilities |
| `cost` | binary cross-entropy, averaged over examples |
| `calculate_gradient` | `dw`, `db` — which way to move each parameter |
| `gradient_descent` | the training loop, logging cost every 100 iterations |
| `predict` | thresholds probabilities at 0.5 into 0/1 labels |
| `train_model` | ties it together and reports train/test accuracy |

Vectors are **features-as-rows, samples-as-columns** (`X` is `(n_features, m)`), which is why `main.py` transposes before training.

## Notes

- Accuracy is measured as plain agreement with the labels. The dataset is imbalanced toward non-defaults, so a high number here isn't the whole story — precision/recall would be the honest next step.
- Hyperparameters (`learning_rate=0.1`, `num_iterations=2000`) are set in `main.py:train` and are worth tuning.
- This is a learning project, not a credit-decisioning system. Don't grant loans with it.
