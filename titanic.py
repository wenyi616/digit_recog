#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:30:43 2019

@author: wenyi
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

trainSet = pd.read_csv('./titanic/train.csv')
testSet = pd.read_csv('./titanic/test.csv')
combine = [trainSet, testSet]

#print(trainSet.columns.values)
#trainSet.describe()

def prepare_data():
    # examine some features (exploring correlations)
    #trainSet[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
    #trainSet[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
    
    #g = sns.FacetGrid(trainSet, col='Survived')
    #g.map(plt.hist, 'Pclass', bins = 10, alpha = .5, color="lightblue")
    #g.add_legend()
    #
    #g2 = sns.FacetGrid(trainSet, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    #g2.map(sns.barplot, 'Sex', 'Fare', alpha=.5)
    #g2.add_legend()
    
    combine = [trainSet, testSet]
    
    # fill in empty values for Embarked feature (using majority vote)
    freq_port = trainSet.Embarked.dropna().mode()[0]
    
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        
        # converted Age feature into a numerical (int)
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
        # converted Age feature into a numerical (int)
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
          
        # fill in empty values for Age feature (using median)
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    #trainSet.head()
    
    # converted Age feature into a categorical (int)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 30), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 4
        dataset.loc[ dataset['Age'] > 60, 'Age']
    #trainSet.head()
    
    
    for dataset in combine:
        # created a new feature (if travel alone) by combining sibsp and parch
        dataset['Family'] = 1 + dataset['SibSp'] + dataset['Parch']  
        dataset['IsAlone'] = 0
        dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
    
        # fill in empty values for Fare feature (using median)
        testSet['Fare'].fillna(testSet['Fare'].dropna().median(), inplace=True)
        
    #trainSet.head()
    #testSet.info()


prepare_data()

# ignore unnecessary features
trainSet = trainSet.drop(['Ticket', 'Cabin'], axis=1)
testSet = testSet.drop(['Ticket', 'Cabin'], axis=1)
trainSet = trainSet.drop(['Parch','SibSp','Family'], axis=1)
testSet = testSet.drop(['Parch','SibSp','Family'], axis=1)
trainSet = trainSet.drop(['Name', 'PassengerId'], axis=1)
testSet = testSet.drop(['Name'], axis=1)
    
trainLabel = trainSet["Survived"]
train = trainSet.drop("Survived", axis=1)
passengerId = testSet['PassengerId']
test  = testSet.drop("PassengerId", axis=1)
    
#trainLabel.shape, train.shape, test.shape
    
    
# use sklearn's logisticRegresssion to train dataset
logreg = LogisticRegression()
logreg.fit(train, trainLabel)
pred = logreg.predict(test)

out = pd.DataFrame(columns=['PassengerId', 'Survived'])
out['PassengerId'] = passengerId
out['Survived'] = pred.astype(int)
out.to_csv('output_2.csv', index=False)

