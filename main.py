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
    test=clean_data.clean_test_data(data_test)
    rf = training.evaluating(train)
    pred = training.prediciton(test, rf)
    training.accuracy(test, pred)
    print(pred)

if __name__ == "__main__":
    main()




