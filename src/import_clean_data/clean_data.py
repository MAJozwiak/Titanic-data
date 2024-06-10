import pandas as pd
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Age'] = data['Age'].fillna(data['Age'].mean().round()).astype(int)
    data['Sex'] = data['Sex'].map({'male':0,'female':1})
    data = data.dropna(subset=['Embarked'])
    data.loc[:, 'Embarked']= data['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
    data.loc[:, 'Embarked']= data['Embarked'].astype(int)
    data = data.drop(["Cabin", "PassengerId", "Name", "Ticket", "Parch", "SibSp"], axis=1)
    print(data)
    return data

def clean_test_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Age'] = data['Age'].fillna(data['Age'].mean().round()).astype(int)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = data.dropna(subset=['Fare'])
    data.loc[:, 'Embarked'] = data['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
    data.loc[:, 'Embarked'] = data['Embarked'].astype(int)
    data = data.drop(["Cabin", "PassengerId", "Name", "Ticket", "Parch", "SibSp"], axis=1)
    print(data)
    return data

