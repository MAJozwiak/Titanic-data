import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Age'] = data['Age'].fillna(data['Age'].mean().round()).astype(int) #Empty Age values filled with mean and then round
    data['Sex'] = data['Sex'].map({'male':0,'female':1}) #Sex values maped from string to int (necessery for further model training)
    data = data.dropna(subset=['Embarked']) #Rows discarded where values in column Embarked are empty
    data = data.dropna(subset=['Fare'])  #Rows discarded where values in column Fare are empty
    data.loc[:, 'Embarked']= data['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int) #Embarked values maped from string to int (necessery for further model training)
    data = data.drop(["Cabin", "PassengerId", "Name", "Ticket", "Parch", "SibSp"], axis=1) #Columns droped
    return data

