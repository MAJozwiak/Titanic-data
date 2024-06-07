from src.import_clean_data import read_data, clean_data
from src.train_model import training

data_train, data_test = read_data.import_data()
train=clean_data.clean_data(data_train)
test=clean_data.clean_test_data(data_test)
rf = training.evaluating(train)
pred=training.prediciton(test, rf)
training.accuracy(test,pred)