#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
file  : prob.py

Define problem class that is used experiments.
"""

import os
import argparse
import numpy as np
import numpy.linalg as la
from scipy.io import savemat, loadmat

parser = argparse.ArgumentParser()
parser.add_argument(
    '-M', '--M', type=int, default=250,
    help="Dimension of measurements.")
parser.add_argument(
    '-N', '--N', type=int, default=500,
    help="Dimension of sparse codes.")
parser.add_argument(
    '-L', '--L', type=int, default=0,
    help="Number of samples for validation (deprecated. please use default).")
parser.add_argument(
    '-P', '--pnz', type=float, default=0.1,
    help="Percent of nonzero entries in sparse codes.")
parser.add_argument(
    '-S', '--SNR', type=str, default='inf',
    help="Strength of noises in measurements.")
parser.add_argument(
    '-C', '--con_num', type=float, default=0.0,
    help="Condition number of measurement matrix. 0 for no modification on condition number.")

def random_A(M, N, con_num=0, col_normalized=True):
    """
    Randomly sample measurement matrix A.
    Curruently I sample A from i.i.d Gaussian distribution with 1./M variance and
    normalize columns.
    TODO: check assumptions on measurement matrix A referring to Wotao Yin's Bregman
    ISS paper.

    :M: integer, dimension of measurement y: 250
    :N: integer, dimension of sparse code x: 500
    :col_normalized:
        boolean, indicating whether normalize columns, default to True
    :returns:
        A: numpy array of shape (M, N)

    """
    A = np.random.normal( scale=1.0/np.sqrt(M), size=(M,N) ).astype(np.float32)
    if con_num > 0:
        U, _, V = la.svd (A, full_matrices=False)
        s = np.logspace (0, np.log10 (1 / con_num), M)
        A = np.dot (U * (s * np.sqrt(N) / la.norm(s)), V).astype (np.float32)
    if col_normalized:
        A = A / np.sqrt(np.sum(np.square(A), axis=0, keepdims=True)) 
    return A
def measure (A, x, SNR=np.inf):
    """
    Measure sparse code x with matrix A and return the measurement.
    TODO:
      Only consider noiseless setting now.
    """
    y   = np.matmul (A, x)
    std = np.std (y, axis=0) * np.power (10.0, -SNR/20.0)  # 每一列的标准差
    std = np.maximum (std, 10e-50)
    noise = np.random.normal (size=y.shape , scale=std).astype (np.float32)

    return y + noise

def gen_samples(A, N,L, pnz=0.1, SNR=np.inf, probability=0.1):
    """
    Generate samples (y, x) in current problem setting.
    TODO:
    - About how to generate sparse code x, need to refer to Wotao' paper
      about the strength of signal x.
      Here I just use standard Gaussian.
    """
    bernoulli = np.random.uniform (size=(N, L)) <= probability
    bernoulli = bernoulli.astype (np.float32)
    x = bernoulli * np.random.normal (size=(N, L)).\
                        astype(np.float32)

    y = measure (A, x, SNR)
    return y, x


'''
generate data:
    A
    x
    y
'''
M = 250
N = 500
Ltrain = 10000
Ltest = 1000
A = random_A(M, N, con_num=0, col_normalized=True)
ytrain,xtrain = gen_samples(A,N,Ltrain)
ytest,xtest = gen_samples(A,N,Ltest)

np.save('./data/Amatrix.npy',A)
np.save('./data/ytrain.npy',ytrain)
np.save("./data/xtrain.npy",xtrain)
np.save('./data/ytest.npy',ytest)
np.save("./data/xtest.npy",xtest)



