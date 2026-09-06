"""Trains the loan-default model, then interactively predicts on user input.

Run it, wait for training to finish, then answer the prompts for as many
loan applications as you like (Ctrl+C or Ctrl+D to stop).
"""

import numpy as np
from src.extraction import extract_data
from src.fetch import fetch_dataset
from src.model import LogisticRegression
from src.parse import parse_loan_application
from src.primitives import DefaultOnFile, LoanGrade, LoanIntent, PersonHomeOwnership

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]


def to_feature_vector(row, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Turn one row (a dataset row, or a parsed application) into a feature vector.

    Numeric fields are standardized with `mean`/`std` from the training set,
    so a single applicant's raw values line up with the scale the model was
    trained on; without this, large raw values (e.g. income) overflow the
    sigmoid during training.
    """
    numeric = np.array([row[field] for field in NUMERIC_FEATURES], dtype=float)
    numeric = (numeric - mean) / std

    home_ownership = PersonHomeOwnership(row["person_home_ownership"]).one_hot()
    intent = LoanIntent(row["loan_intent"]).one_hot()
    grade = LoanGrade(row["loan_grade"]).one_hot()
    default_on_file = DefaultOnFile(row["cb_person_default_on_file"]).encode_flag()

    return np.concatenate([numeric, home_ownership, intent, grade, default_on_file])


def train() -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fetch the dataset and train the model on it.

    Returns:
        The trained model, plus the numeric-feature mean/std computed from
        the training set (needed again to encode any new application the
        same way at prediction time).
    """
    df = extract_data(fetch_dataset())

    numeric_raw = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    mean = numeric_raw.mean(axis=0)
    std = numeric_raw.std(axis=0)

    X = np.stack(df.apply(lambda row: to_feature_vector(row, mean, std), axis=1))
    y = df["loan_status"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    # LogisticRegression expects features-as-rows, samples-as-columns.
    X_train, X_test = X[train_idx].T, X[test_idx].T
    Y_train, Y_test = y[train_idx].reshape(1, -1), y[test_idx].reshape(1, -1)

    model = LogisticRegression(learning_rate=0.1, num_iterations=2000)
    model.train_model(X_train, Y_train, X_test, Y_test)

    return model, mean, std


def main():
    print("Training model...\n")

    model, mean, std = train()

    print("\nModel trained. Enter a loan application to get a prediction (Ctrl+C to quit).\n")

    while True:
        try:
            application = parse_loan_application()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")

            break

        features = to_feature_vector(application, mean, std).reshape(-1, 1)

        prediction = model.predict(features)

        if prediction[0, 0] == 1:
            print("Prediction: this loan is likely to DEFAULT - do not grant.\n")
        else:
            print("Prediction: this loan is likely to be repaid - safe to grant.\n")


if __name__ == "__main__":
    main()
