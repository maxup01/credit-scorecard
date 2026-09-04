from pandas import DataFrame

def extract_data(dataset: DataFrame) -> DataFrame:
    return dataset.dropna(subset=["person_emp_length", "loan_int_rate"])