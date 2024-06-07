import pandas as pd
def import_data():
    data_train=pd.read_csv(r"C:\Users\marta\PycharmProjects\Titanic-data\data\Titanic-Dataset.csv")
    data_test=pd.read_csv(r"C:\Users\marta\PycharmProjects\Titanic-data\data\test.csv")
    print(data_train)
    print(data_test)
