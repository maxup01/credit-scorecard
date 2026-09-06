"""Prompts the user for the fields of loan application.

Returns them as a plain dict keyed the same way as a row of the dataset, so
the same feature-vector logic used for training can be reused unchanged at
prediction time.
"""

from src.primitives import DefaultOnFile, LoanGrade, LoanIntent, PersonHomeOwnership


def _prompt_float(label: str) -> float:
    while True:
        raw = input(f"{label}: ").strip()

        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' isn't a number, try again.")


def _prompt_choice(label: str, options: list[str]) -> str:
    choices = "/".join(options)

    while True:
        value = input(f"{label} ({choices}): ").strip().upper()
        if value in options:
            return value

        print(f"  Please enter one of: {choices}")


def parse_loan_application() -> dict:
    """Prompt for one loan application's fields.

    Returns:
        dict: Raw field values, keyed the same way as the dataset's columns,
        ready to be turned into a feature vector by the same code that builds training features.
    """
    return {
        "person_age": _prompt_float("Applicant age"),
        "person_income": _prompt_float("Annual income"),
        "person_home_ownership": _prompt_choice("Home ownership", PersonHomeOwnership.values()),
        "person_emp_length": _prompt_float("Employment length (years)"),
        "loan_intent": _prompt_choice("Loan intent", LoanIntent.values()),
        "loan_grade": _prompt_choice("Loan grade", LoanGrade.values()),
        "loan_amnt": _prompt_float("Loan amount"),
        "loan_int_rate": _prompt_float("Loan interest rate (%)"),
        "loan_percent_income": _prompt_float("Loan amount as a fraction of income"),
        "cb_person_default_on_file": _prompt_choice(
            "Defaulted before", [DefaultOnFile.YES.value, DefaultOnFile.NO.value]
        ),
        "cb_person_cred_hist_length": _prompt_float("Credit history length (years)"),
    }
