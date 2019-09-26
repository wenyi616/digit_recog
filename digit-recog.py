#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:52:16 2019

@author: wenyi
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial import distance
from itertools import combinations 
from sklearn.metrics import confusion_matrix


# extract pixel data (pixel0 tp pixel784)
# extract labels
train_data = np.loadtxt(open("./digit-recognizer/train_2.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1,785), dtype=np.int)
labels = np.loadtxt(open("./digit-recognizer/train_2.csv", "rb"), delimiter=",", skiprows=1, usecols=range(0,1), dtype=np.int)
test_data = np.loadtxt(open("./digit-recognizer/test.csv", "rb"), delimiter=",", skiprows=1, usecols=range(0,784), dtype=np.int)

all_digit_index = [1,0,16,7,3,8,21,6,10,11]

print(train_data.shape)
print(test_data.shape)

# extract index of zeros and ones  
zeros = []
ones = []
for i in range(0,len(train_data)):
    if labels[i] == 0:
        zeros.append(i)
    if labels[i] == 1:
        ones.append(i)
            
# a function to display an MNIST digit
def display_digit(index):
    plt.figure(figsize=(1, 1))
    ax = plt.subplot(1,1,1)
    ax.imshow(train_data[index].reshape(28,28), cmap='gray')
    
# a function to display all MNIST digits, from 0 to 9
def display_digits():
#    digit_index = [1,0,16,7,3,8,21,6,10,11]
    plt.figure(figsize=(10, 5))
    for i in range(10):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(train_data[all_digit_index[i]].reshape(28,28), cmap='gray')
            
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
    
# find distance given two indices of train_data
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
    min_d = float("inf")
    match = index
    
    # find best match
    for i in range(len(train_data)):
        d = find_distance(i,index)

        if (d < min_d and i!=index):
            match = i
            min_d = d      
#    print("best match for row %d is at row %d, dist=%f, %d,%d" % (index, match, min_d,labels[index],labels[match]))
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
        genuine_dist.append(d)
    
    for i in comb_ones:
        d = find_distance(i[0],i[1])
        genuine_dist.append(d)    
          
    # pairwise distances for all imposter matches
    for i in zeros:
        for j in ones:
            d = find_distance(i,j)
            impostor_dist.append(d) 

#    print(len(genuine_dist),len(impostor_dist))
#    print(genuine_dist)
#    print(impostor_dist)


    # Plot histograms of the genuine and impostor distances on the same set of axes.
    plt.figure(figsize=(10, 5))
    custom_bins = range(500, 4000, 50)
    
    plt.hist(genuine_dist, alpha=0.5, density=True, bins=custom_bins, label='Genuine', color="lightblue",width=50)
    plt.hist(impostor_dist, alpha=0.5, density=True, bins=custom_bins, label='Imposter', color="lightgrey",width=50)
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
    
    print(x,y)
    # mark ERR on the curve
    plt.plot([x], [y], marker='o', markersize=5, color="grey")
    plt.show()
   
# **************************** Test Part 1 ****************************
#display_digits()
#display_distribution()
# match_digits()
#zeros_ones()
    

#1.g
def KNN(test, train_data, train_labels, k):    
    output = []
    
    print("Total test case %d" % len(test))
    for i in range(len(test)):
        test_case = test[i]
        
        if (i % 500 == 0):
            print("*** current test case: %d" % i)
            
#        print("*** current test case: %d" % i)
#        plot_test(test[i])
        
        # for each test case, iterate in the training data and find all distances
        ds = []
        
        for i in range(len(train_data)):
            d = np.linalg.norm(train_data[i] - test_case)
            ds.append((i, d))

        ds.sort(key=lambda elem: elem[1])
        
        # find k neighbors
        neighbors = []
        for i in range(k):
            neighbors.append(ds[i][0])
        
        # find freq
        freq = {}
        for i in range(len(neighbors)):
            response = labels[neighbors[i]]
            if response in freq:
                freq[response] += 1
            else:
                freq[response] = 1
        
        freq = sorted(freq.items(), key=lambda elem: elem[1], reverse=True)
        
        # Classify using majority vote among the k neighbors
        match = freq[0][0]
            
#        print(match)
        output.append(match)
                
    return output

def plot_test(pixels): 
    plt.figure(figsize=(1, 1))
    ax = plt.subplot(1,1,1)
    ax.imshow(pixels.reshape(28,28), cmap='gray')
    plt.show()

#compare_res_n_label(KNN(test_data, train_data, labels, 10),[2,0,9,0,3,7,0,3,0,3,5,7] )
  
# compare two lists and calculate accuracy
def compare_res_n_label(results,labels):
    count = 0
    miss = 0
    for i in range(len(results)):
        if results[i] == labels[i]:
            count = count + 1
        else:
            miss = miss + 1
            
    print(count,miss,count+miss)
    
    return float(count) / float(count+miss)
    
    
# Using the training data for all digits
# perform 3 fold cross-validation on your K-NN classifier
# report your average accuracy
def threefold_cross_valid():

    trainings = len(train_data)
    # divide up your data into N=3 buckets
    
    # train = 1+2, test = 3
    t1 = train_data[0:int(trainings*2/3)]   #training-data
    tl1 = labels[0:int(trainings*2/3)]      #training-label
    v1 = train_data[int(trainings*2/3):]    #validation-data
    vl1 = labels[int(trainings*2/3):]       #validation-labels
    
    # train = 1+3, test = 2
    t21 = train_data[0:int(trainings*2/3)]
    t22 = train_data[int(trainings*2/3):]
    tl21 = labels[0:int(trainings*2/3)]
    tl22 = labels[int(trainings*2/3):] 
    
    t2 = np.concatenate((t21, t22), axis=0)
    tl2 = np.concatenate((tl21, tl22), axis=0)
    v2 = train_data[int(trainings/3):int(trainings*2/3)]    #validation-data
    vl2 = labels[int(trainings/3):int(trainings*2/3)]       #validation-labels

    
    # train = 2+3, test = 1
    t3 = train_data   #training-data
    tl3 = labels[int(trainings/3):0]  
    v3 = train_data[:int(trainings/3)]    #validation-data
    vl3 = labels[:int(trainings/3)]       #validation-labels
    
    
    results_1 = KNN(v1, t1, tl1, 3)
    results_2 = KNN(v2, t2, tl2, 3)
    results_3 = KNN(v3, t3, tl3, 3)
    
    a1 = compare_res_n_label(results_1,vl1)
    a2 = compare_res_n_label(results_2,vl2)
    a3 = compare_res_n_label(results_3,vl3)
    
    print(confusion_matrix(results_1, vl1))
    print("")
    print(confusion_matrix(results_2, vl2))        
    print("")
    print(confusion_matrix(results_3, vl3))
    
    print(a1,a2,a3)
    print("The average accuracy is %.2f" % ((a1+a2+a3)/3))



def output():
    
    l = KNN(test_data, train_data, labels, 3)
    out = ["ImageId,Label"]
    
    for i in range(len(test_data)):
        s = "%d,%d" % (i+1,l[i])
        out.append(s)
    
    MyFile=open('output.csv','w')
    for element in out:
        print >>MyFile, element
    MyFile.close()

# **************************** Test Part 2 ****************************
#threefold_cross_valid()
#KNN(test_data, train_data, labels, 10)  
output()