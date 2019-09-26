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

train_df = pd.read_csv('./titanic/train.csv')
test_df = pd.read_csv('./titanic/test.csv')
combine = [train_df, test_df]

#print(train_df.columns.values)
#train_df.describe()

def prepare_data():
    # examine some features (exploring correlations)
    #train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
    #train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
    
    #g = sns.FacetGrid(train_df, col='Survived')
    #g.map(plt.hist, 'Pclass', bins = 10, alpha = .5, color="lightblue")
    #g.add_legend()
    #
    #g2 = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    #g2.map(sns.barplot, 'Sex', 'Fare', alpha=.5)
    #g2.add_legend()
    
    combine = [train_df, test_df]
    
    # fill in empty values for Embarked feature (using majority vote)
    freq_port = train_df.Embarked.dropna().mode()[0]
    
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        
    # converted Age feature into a numerical (int)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        
    
    # converted Age feature into a numerical (int)
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        
        
    # fill in empty values for Age feature (using median)
    for dataset in combine:
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    #train_df.head()
    
    # converted Age feature into a categorical (int)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 30), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 4
        dataset.loc[ dataset['Age'] > 60, 'Age']
    #train_df.head()
    
    
    # created a new feature (if travel alone) by combining sibsp and parch
    for dataset in combine:
        dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
    
    # fill in empty values for Fare feature (using median)
    for dataset in combine:
        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
        
    #train_df.head()
    #test_df.info()



prepare_data()

# ignore unnecessary features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
train_df = train_df.drop(['Parch','SibSp','Family'], axis=1)
test_df = test_df.drop(['Parch','SibSp','Family'], axis=1)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
    
train_label = train_df["Survived"]
train = train_df.drop("Survived", axis=1)
passengerId = test_df['PassengerId']
test  = test_df.drop("PassengerId", axis=1)
    
#train_label.shape, train.shape, test.shape
    
    
# use sklearn's logisticRegresssion to train dataset
logreg = LogisticRegression()
logreg.fit(train, train_label)
pred = logreg.predict(test)

out = pd.DataFrame(columns=['PassengerId', 'Survived'])
out['PassengerId'] = passengerId
out['Survived'] = pred.astype(int)
out.to_csv('output_2.csv', index=False)

