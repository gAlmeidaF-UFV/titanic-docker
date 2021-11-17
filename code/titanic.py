import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#configuring random seed
np.random.seed(0)

#importing datasets
test_path = './dataset/test.csv'
train_path = './dataset/train.csv'

test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

#selecting util features
train_num = train.loc[:, ['PassengerId','Survived','Age', 'SibSp', 'Parch', 'Fare', 'Pclass']]
train_num = train_num.dropna(axis = 0)
test_num = test.loc[:, ['PassengerId','Age', 'SibSp', 'Parch', 'Fare', 'Pclass']]
test_num = test_num.fillna(0)

#creating dataframe
train_age = train_num.loc[:,['Survived', 'Age']]

#creating new feature
bins = np.arange(0,90,10)
labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']

train_age['AgeGroup'] = pd.cut(train_num['Age'], bins, labels = labels)
train_age.head()

#defining target and features that will be used on trainnig 
X = train_num.drop(['PassengerId','Survived'], axis = 1)
y = train_num['Survived']

#spliting dataset into train and validation, 20% used for validation
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training model
randomforest = RandomForestClassifier()
randomforest.fit(train_X, train_y)
preds = randomforest.predict(val_X)

#building predictions
index = test_num['PassengerId']
prediction = randomforest.predict(test_num.drop('PassengerId', axis = 1))

#building file from predictions
output = pd.DataFrame({'PassengerId':index, 'Survived': prediction})
output.to_csv('./output/submission.csv', index = False)