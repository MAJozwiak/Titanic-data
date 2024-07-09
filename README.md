# Machine Learning Project: Binary Classification Using the Titanic Dataset
- **Link to Titanic dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data**

The goal of this project is to build and compare three different binary classification models to predict whether a passenger survived the Titanic disaster.
The classic Titanic dataset, which includes various features of the passengers, was used for this analysis. The project includes stages of exploratory data analysis, feature engineering, model training, to evaluation.
I've implemented models: Random Forrest, Support Vector Machine, Decision Tree.
## Binary Classification Models definitions
- **Support Vector Machine (SVM):**
SVM finds the optimal hyperplane which maximizes the margin between different classes in the feature space.
- **DecisionTreeClassifier:**
DecisionTree makes decisions based on a series of if-then rules derived from the features. It splits the data into subsets based on the most significant feature at each node, creating a tree structure to make predictions.
- **RandomForestClassifier:**
RandomForest combines multiple decision trees. It uses bootstrap aggregation and random feature selection to create diverse trees.
## Description of files
- ***main.py*** - file where function calls  are located,
- ***plots.ipynb*** - notebook that contains data analysis,
- ***read.py*** - file that contains importing and reading training and testing datasets,
- ***clean.py*** - file that cleans the training and testing data,
- ***training.py*** - file that contains the implementation of three predictive models and a function responsible for saving the results to a score.csv file,
- ***config.yaml*** - config file that contains paths, to csv files (datasets and score's file),
- ***test.csv*** - testing data,
- ***Titanic-Dataset.csv*** - training data.
## Metrics
The results of all three predictions have been saved to the score.csv file. Four different metrics were calculated for each model:
- ***Accuracy***: the percentage of correct classifications,
- ***Precision***: the percentage of correctly predicted positive cases,
- ***Recall***: the percentage of actual positive cases correctly identified,
- ***F1 Score***: the harmonic mean of precision and recall.
## Result
Based on the final results, it is possible to create a ranking of implementations:
- ***1.SVM*** 
- ***2.DecisionTree*** - Better result than RandomForest obtained through GridSearch help.
- ***3.RandomForest***
