from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier()
    parameters = {'n_estimators': [100], 'max_depth': [2]}
    clf = GridSearchCV(rf, parameters, cv=5)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def svm_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> svm.SVC:
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf


def decision_tree(X_train: pd.DataFrame, y_train: pd.DataFrame) -> DecisionTreeClassifier:
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


def score(clf, X_test: pd.DataFrame, y_test: pd.DataFrame, open_file: str) -> None:
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    clf_name = clf.__class__.__name__
    with open(open_file, 'a') as file:
        file.write(
            f"Classifier: {clf_name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1_score: {f1}\n")
