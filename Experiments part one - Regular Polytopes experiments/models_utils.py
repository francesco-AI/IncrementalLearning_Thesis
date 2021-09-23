"""
##############################################################################
##############################################################################
########               Artificial Intelligence                ###############
########                      Thesy                            ###############
########         Prof. Capobianco - Prof. Lo Monaco            ###############
########                                                       ###############
########                                                       ###############
########             Students: FRANCESCO CASSINI               ###############
########             Sapienza IDs:       785771                ###############
########     Master in Roboics and Artificial Intelligence     ###############
##############################################################################
##############################################################################



import torch
import torch.nn as nn

import math
from torch.autograd import Variable
from scipy.linalg import hadamard
import numpy as np


#######################################################################################################
################   Hadamard       ##################################################################
#######################################################################################################

# Example of Hadamard 
# tensor([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#         [ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
#         [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1],
#         [ 1, -1, -1,  1,  1, -1, -1,  1,  1, -1],
#         [ 1,  1,  1,  1, -1, -1, -1, -1,  1,  1],
#         [ 1, -1,  1, -1, -1,  1, -1,  1,  1, -1],
#         [ 1,  1, -1, -1, -1, -1,  1,  1,  1,  1],
#         [ 1, -1, -1,  1, -1,  1,  1, -1,  1, -1],
#         [ 1,  1,  1,  1,  1,  1,  1,  1, -1, -1],
#         [ 1, -1,  1, -1,  1, -1,  1, -1, -1,  1]])

#######################################################################################################

def hadamard_matrix(features):
    # number of dimension matrix is given by log2 of the number of classes
    # with math.ceil we approximate at first near integer
    dimension = 2**(math.ceil(math.log2(features)))
    # to compute hadamard we use the standard math library to get a square hadamard matrix
    hadamard_matrix = torch.tensor(hadamard(dimension))
    return hadamard_matrix





#######################################################################################################
################   d-Simplex       ##################################################################
#######################################################################################################

# Example of  d-Cube
# tensor([[ 0.9740, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801,  0.9740, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801,  0.9740, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801,  0.9740, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801, -0.0801,  0.9740, -0.0801, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801, -0.0801, -0.0801,  0.9740, -0.0801, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801,  0.9740, -0.0801, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801,  0.9740, -0.0801, -0.3333],
#         [-0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801, -0.0801,  0.9740, -0.3333]])



#######################################################################################################



def dsimplex_matrix(num_classes=10):
    in_features = num_classes - 1
    out_features = num_classes

    weigths_matrix = torch.zeros([in_features, in_features + 1])
    for j in range(0, in_features):
        weigths_matrix[j, j] = 1.0
    #value correspond to alfa value: see the paper of Pernici at bottom of page 4
    value = (1.0 - np.sqrt(float(1 + in_features))) / float(in_features)
    # in diagonal of matrix put the value
    for i in range(0, in_features):
        weigths_matrix[i, in_features] = value

    # now we want to center our matrix
    # we have to get the center matrix and subtrac to original matrix
    center_matrix = torch.zeros(in_features)
    for i in range(0, in_features):
        center_value = 0.0
        for j in range(0, in_features + 1):
            center_value = center_value + weigths_matrix[i, j]
        center_matrix[i] = center_value / float(in_features + 1)

    for j in range(0, in_features + 1):
        for i in range(0, in_features):
            weigths_matrix[i, j] = weigths_matrix[i, j] - center_matrix[i]

    #  At the end we have to normalize value
    center_value = 0.0
    for i in range(0, in_features):
        center_value = center_value + weigths_matrix[i, 0] ** 2
    center_value = np.sqrt(center_value)

    for j in range(0, in_features + 1):
        for i in range(0, in_features):
            weigths_matrix[i, j] = weigths_matrix[i, j] / center_value

    return weigths_matrix










#######################################################################################################
################   d-Orthoplex       ##################################################################
#######################################################################################################

# Example of  d-ORTHOPLEX 
# tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.]])

#######################################################################################################


def dorthoplex_matrix(num_classes):
    in_features = num_classes // 2
    out_features = num_classes
    #at beginning we create a weight_matrix full of zero
    weights_matrix = torch.zeros(out_features, in_features)
    # in case of d-orthoplex, the row are vectors of (+1,0,0,0,..) (-1,0,0,0,0) (0,+1,0,0,0,..) (0,-1,0,0,0,0)
    # so we create a for cycle to alternate +1 and -1
    for row in range(out_features):
        col = row // 2
        weights_matrix[row, col] = (-1) ** row
    return weights_matrix




#######################################################################################################
################   d-Cube       ##################################################################
#######################################################################################################

# Example of  d-Cube
# tensor([[-0.5000, -0.5000, -0.5000, -0.5000],
#         [-0.5000, -0.5000, -0.5000,  0.5000],
#         [-0.5000, -0.5000,  0.5000, -0.5000],
#         [-0.5000, -0.5000,  0.5000,  0.5000],
#         [-0.5000,  0.5000, -0.5000, -0.5000],
#         [-0.5000,  0.5000, -0.5000,  0.5000],
#         [-0.5000,  0.5000,  0.5000, -0.5000],
#         [-0.5000,  0.5000,  0.5000,  0.5000],
#         [ 0.5000, -0.5000, -0.5000, -0.5000],
#         [ 0.5000, -0.5000, -0.5000,  0.5000],
#         [ 0.5000, -0.5000,  0.5000, -0.5000],
#         [ 0.5000, -0.5000,  0.5000,  0.5000],
#         [ 0.5000,  0.5000, -0.5000, -0.5000],
#         [ 0.5000,  0.5000, -0.5000,  0.5000],
#         [ 0.5000,  0.5000,  0.5000, -0.5000],
#         [ 0.5000,  0.5000,  0.5000,  0.5000]], dtype=torch.float64)


#######################################################################################################


def dcube_matrix(num_classes):
    in_features = math.ceil(math.log2(num_classes))
    out_features = num_classes
    #at beginning we create a weight_matrix full of zero
    weights_matrix = torch.zeros(out_features, in_features)
    max_len = len(str(bin(out_features-1)))-2
    for row in range(out_features):
        binary = str(bin(row))[2:]
        binary = '0'*(max_len - len(binary)) + binary
        for col in range(len(binary)):
            weights_matrix[row,col] = -0.5 + int(binary[col])
    return weights_matrix

