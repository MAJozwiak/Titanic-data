from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def evaluating(data):
    rf = RandomForestClassifier()
    X = data.drop('Survived', axis=1)
    Y = data['Survived']
    rf.fit(X,Y)
    return rf
def prediciton(data,rf):
    X_test=data.drop('Survived', axis=1)
    y_pred = rf.predict(X_test)
    return y_pred
def accuracy(data, y_pred):
    y_test = data['Survived']
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)



