import pandas as pd
from pandas import DataFrame
from typing import Tuple


def import_data(train_path: str, test_path: str) -> Tuple[DataFrame, DataFrame]:
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    return data_train, data_test
