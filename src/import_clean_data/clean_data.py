
def clean_data(data):
    data['Age']=data['Age'].fillna(data['Age'].mean().round())
    data["Age"]=data["Age"].astype(int)
    data['Sex'] = data['Sex'].map({'male':0,'female':1})
    data = data.dropna(subset=['Embarked'])
    data.loc[:, 'Embarked']= data['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
    data.loc[:, 'Embarked']= data['Embarked'].astype(int)
    data=data.drop("Cabin", axis=1)
    data=data.drop("PassengerId", axis=1)
    data=data.drop("Name", axis=1)
    data=data.drop("Ticket", axis=1)
    data=data.drop("Parch", axis=1)
    data=data.drop("SibSp", axis=1)
    print(data)
    return data

def clean_test_data(data):
    data['Age'] = data['Age'].fillna(data['Age'].mean().round())
    data["Age"] = data["Age"].astype(int)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = data.dropna(subset=['Fare'])
    data.loc[:, 'Embarked'] = data['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
    data.loc[:, 'Embarked'] = data['Embarked'].astype(int)
    data = data.drop("Cabin", axis=1)
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)
    data = data.drop("Parch", axis=1)
    data = data.drop("SibSp", axis=1)
    print(data)
    return data

