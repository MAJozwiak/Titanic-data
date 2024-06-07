from src.import_clean_data import read_data, clean_data

#data import
data_train,data_test=read_data.import_data()
clean_data.clean_data(data_train)

