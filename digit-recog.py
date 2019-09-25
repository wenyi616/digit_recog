#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:52:16 2019

@author: wenyi
"""

import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
#from scipy.spatial import distance
from itertools import combinations 
from sklearn.metrics import confusion_matrix

# extract pixel data (pixel0 tp pixel784)
# extract labels
train_data = np.loadtxt(open("./digit-recognizer/train_1.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1,785), dtype=np.int)
labels = np.loadtxt(open("./digit-recognizer/train_1.csv", "rb"), delimiter=",", skiprows=1, usecols=range(0,1), dtype=np.int)

print(train_data.shape)
#print(train_data)

# extract index of zeros and ones  
zeros = []
ones = []
for i in range(0,len(train_data)):
    if labels[i] == 0:
        zeros.append(i)
    if labels[i] == 1:
        ones.append(i)
        
print(len(zeros))
print(len(ones))

     
            
# a function to display an MNIST digit
def display_digit(index):
    plt.figure(figsize=(1, 1))
    ax = plt.subplot(1,1,1)
    ax.imshow(train_data[index].reshape(28,28), cmap='gray')
    
# a function to display all MNIST digits, from 0 to 9
def display_digits():
    digit_index = [1,0,16,7,3,8,21,6,10,11]
    plt.figure(figsize=(10, 5))
    for i in range(10):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(train_data[digit_index[i]].reshape(28,28), cmap='gray')
            
# display a normalized histogram of digit counts
def display_distribution():
    plt.figure(figsize=(10, 5))
    plt.hist(labels, range(11), alpha=0.5, density=True, color='grey', width=0.8)
    plt.xticks(np.arange(0, 10, 1))
    plt.ylim(0, 0.2)
    plt.grid(True)

    plt.xlabel("digits")
    plt.ylabel("distribution")
    
# Problem 1.d
def match_digits():
    digit_index = [1,0,16,7,3,8,21,6,10,11]
    for i in range(10):
        display_match(digit_index[i])
    

def find_distance(index1,index2):
#    return distance.euclidean(train_data[index1], train_data[index2])
    return np.linalg.norm(train_data[index1] - train_data[index2])

def display_match(index):
    plt.figure(figsize=(2, 1))

    ax = plt.subplot(1,2,1)
    ax.imshow(train_data[index].reshape(28,28), cmap='gray')

    ax = plt.subplot(1,2,2)
    match = match_by_index(index)
    ax.imshow(train_data[match].reshape(28,28), cmap='gray')

    if (labels[index] != labels[match]):
        ax.set_title("*",fontsize=18)
  
def match_by_index(index):
#    print("looking for digit at row %d, which is a %d" % (index, labels[index]))    
    min_d = float("inf")
    match = index
    
    # find best match
    for i in range(len(train_data)):
        d = find_distance(i,index)

        if (d < min_d and i!=index):
            match = i
            min_d = d      
#    print("best match for row %d is at row %d, dist=%f" % (index, match, min_d))
    return match


# problem 1.e
def zeros_ones():
    genuine_dist = []
    impostor_dist = []
    
    # pairwise distances for all genuine matches
    comb_zeros = list(combinations(zeros, 2))
    comb_ones = list(combinations(ones, 2))

    for i in comb_zeros:
        d = find_distance(i[0],i[1])
#        print("index %d, index %d, label %d, label %d, d = %f" % (i[0], i[1], labels[i[0]], labels[i[1]], d))
        genuine_dist.append(d)
    
    for i in comb_ones:
        d = find_distance(i[0],i[1])
#        print("index %d, index %d, label %d, label %d, d = %f" % (i[0], i[1], labels[i[0]], labels[i[1]], d))
        genuine_dist.append(d)    
          
    # pairwise distances for all imposter matches
    for i in zeros:
        for j in ones:
            d = find_distance(i,j)
#            print("index %d, index %d, label %d, label %d, d = %f" % (i, j, labels[i], labels[j], d))
            impostor_dist.append(d) 

    print(len(genuine_dist),len(impostor_dist))
#    print(genuine_dist)
#    print(impostor_dist)


    # Plot histograms of the genuine and impostor distances on the same set of axes.
    plt.figure(figsize=(10, 5))
    custom_bins = range(500, 4000, 50)
    
    plt.hist(genuine_dist, alpha=0.5, bins=custom_bins, label='Genuine', color="lightblue",width=50)
    plt.hist(impostor_dist, alpha=0.5, bins=custom_bins, label='Imposter', color="lightgrey",width=50)
    plt.legend()
    plt.xticks(np.arange(0, 4500, 500))
    
    plt.grid(True)
    plt.show()

    display_ROC(genuine_dist,impostor_dist)
    


# probelm 1.f
# Generate an ROC curve from the above sets of distances
def display_ROC(genuine,impostor):
    TPRs = []
    FPRs = []
    
    genuine.sort()
    impostor.sort()
    
    thresholds = range(0, len(impostor), int(len(impostor))/100)
    
    for i in thresholds:
        fpr = (impostor < impostor[i]).sum()/float(len(impostor))
        tpr = (genuine < impostor[i]).sum()/float(len(genuine))
        
        TPRs.append(tpr)
        FPRs.append(fpr)
#        print(tpr,fpr)
     
    plt.figure(figsize=(10, 5))
    plt.plot(FPRs,TPRs, color="lightblue")

    plt.title("ROC Curve")
    plt.xlabel("FPR: False Positive")
    plt.ylabel("TPR: True Positive")  

    tpr_index = np.abs(np.add(FPRs,TPRs)-1).argmin()
    
    x = FPRs[tpr_index] #err
    y = TPRs[tpr_index]
    
#    print(x,y)
    # mark ERR on the curve
    plt.plot([x], [y], marker='o', markersize=5, color="grey")
    plt.show()
   
    
#display_digits()
#display_distribution()
# match_digits()
zeros_ones()