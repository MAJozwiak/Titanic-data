import click
import yaml
from src.import_clean_data import read_data, clean_data
from src.train_model import training

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.option('--config-path', default='config.yaml', help='Path to the .yaml file')
def main(config_path):
    config = load_config(config_path)
    train_path = config['paths']['train_data']
    test_path = config['paths']['test_data']

    data_train, data_test = read_data.import_data(train_path, test_path)
    train=clean_data.clean_data(data_train)
    test=clean_data.clean_data(data_test)
    X_train=train.drop('Survived', axis=1)
    Y_train=train['Survived']
    X_test=test.drop('Survived', axis=1)
    Y_test=test['Survived']
    clf_forest=training.random_forest(X_train,Y_train)
    clf_svm=training.svm_model(X_train, Y_train)
    clf_decision_tree=training.decision_tree(X_train, Y_train)
    print("Random Forest")
    training.score(clf_forest,X_test,Y_test)
    print("SVM")
    training.score(clf_svm, X_test, Y_test)
    print("Decision Tree")
    training.score(clf_decision_tree, X_test, Y_test)



if __name__ == "__main__":
    main()




