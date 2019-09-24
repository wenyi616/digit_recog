#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:52:16 2019

@author: wenyi
"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance


# extract pixel data (pixel0 tp pixel784)
# extract labels
train_data = np.loadtxt(open("./digit-recognizer/train.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1,785), dtype=np.int)
labels = np.loadtxt(open("./digit-recognizer/train.csv", "rb"), delimiter=",", skiprows=1, usecols=range(0,1), dtype=np.int)

print(train_data.ndim)
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

print(zeros)
print(len(zeros))
#print(ones)

            
            
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
    plt.hist(labels, range(11), alpha=0.5, density=True, color='black', width=0.8)
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
    distances = []
    
    for i in range(len(train_data)):
        d = find_distance(i,index)
#        print(d)
        distances.append(d)
        if (d < min_d and i!=index):
            match = i
            min_d = d      
#    print("best match is at row %d, which is a %d" % (match, labels[match]))
    return match

    
# problem 1.e
def zero_one_match():
    genuine_count = 0
    impostor_count = 0
    genuine_dist = []
    impostor_dist = []

    for i in zeros:
        result = (labels[i] == labels[match_by_index(i)])

        if result:
            genuine_count += 1
            genuine_dist.append(find_distance(match_by_index(i),i))
        else:
            impostor_count += 1
            impostor_dist.append(find_distance(match_by_index(i),i))

            
    print(genuine_count,impostor_count)
    print(genuine_dist,impostor_dist)

    # Plot histograms of the genuine and impostor distances on the same set of axes.
    plt.figure(figsize=(10, 5))
    plt.hist(genuine_dist, alpha=0.5, density=True, label='Genuine', width=10)
    plt.hist(impostor_dist, alpha=0.5, density=True, label='Imposter', width=10)
    plt.legend()

#    plt.ylim(0, 0.05)
    plt.grid(True)

    plt.xlabel("dist")
    plt.ylabel("distribution")
            

    
#display_digits()
#display_distribution()
#match_digits()
zero_one_match()