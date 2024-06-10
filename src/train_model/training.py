from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
def random_forest(X_train,y_train):
    rf = RandomForestClassifier()
    parameters = {'n_estimators': [100], 'max_depth': [2]}
    clf = GridSearchCV(rf, parameters, cv=5)
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def svm_model(X_train,y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

def decision_tree(X_train,y_train):
    clf = DecisionTreeClassifier()
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def score(clf,X_test,y_test,open_file='score.csv'):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy)
    with open(open_file, 'a') as file:
        file.write(f"Accuracy: {accuracy}\n")
