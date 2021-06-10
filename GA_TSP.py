# -*— coding: utf-8 -*_
# @Time    : 2021/5/15 下午6:53
# @Author  : Algo
# @Site    :
# @File    : GA_TSP.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt

# Hyper parameters
POP_SIZE = 30
P_C = 0.75
P_M = 0.05
MAX_GEN = 2000
LEN_EN = 30
GEN = 0
MAX_FIT = []
MEAN_FIT = []
POP = np.random.randint(0, 2, (POP_SIZE, LEN_EN))

def fitness(x):
