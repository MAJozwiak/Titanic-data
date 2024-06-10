from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def evaluating(data):
    rf = RandomForestClassifier()
    parameters = {'n_estimators': [100], 'max_depth': [2]}
    clf = GridSearchCV(rf, parameters, cv=5)
    X = data.drop('Survived', axis=1)
    Y = data['Survived']
    clf.fit(X, Y)
    return clf
def prediciton(data,clf):
    X_test=data.drop('Survived', axis=1)
    y_pred = clf.predict(X_test)
    return y_pred
def accuracy(data, y_pred):
    y_test = data['Survived']
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)



