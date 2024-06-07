import pandas as pd
def import_data():
    data=pd.read_csv(r"C:\Users\marta\PycharmProjects\Titanic-data\data\Titanic-Dataset.csv")
    print(data)
    return data
