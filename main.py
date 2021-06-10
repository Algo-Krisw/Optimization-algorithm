# -*— coding: utf-8 -*_
# @Time    : 2021/5/1 5:19 下午
# @Author  : Algo
# @Site    :
# @File    : main.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt

# Hyper parameters
POP_SIZE = 50
P_C = 0.75
P_M = 0.05
MAX_GEN = 200
LEN_EN = 22
GEN = 0
VALUE_MAX = 2
VALUE_MIN = -1
VALUE_MARGIN = VALUE_MAX - VALUE_MIN
MAX_FIT = []
MEAN_FIT = []
POP = np.random.randint(0, 2, (POP_SIZE, LEN_EN))


def fitness(x):
    return x * np.sin(10 * np.pi * x) + 2.0


def encoder(pop):
    x_array = np.zeros((POP_SIZE, 1))
    for i in range(POP_SIZE):
        x = 0
        for j in range(LEN_EN):
            if pop[i][::-1][j]:
                x += np.power(2, j)
        x_array[i] = VALUE_MARGIN * x / (np.power(2, LEN_EN) - 1) + VALUE_MIN
    return x_array


def select(fit, pop):
    new_pop = np.zeros((POP_SIZE, LEN_EN))
    choice_prob = fit / np.sum(fit)
    choice_index = np.arange(0, POP_SIZE)
    index_choice = np.random.choice(choice_index.flatten(), POP_SIZE, p=choice_prob.flatten())
    for i in range(POP_SIZE):
        new_pop[i] = pop[index_choice[i]]
    new_pop[0] = pop[np.argmax(fit)]
    return new_pop


def crossover(pop):
    new_pop = np.zeros((POP_SIZE, LEN_EN))
    for i in range(0, POP_SIZE, 2):
        pc = np.random.rand()
        if pc <= P_C:
            new_child1 = pop[i].copy()
            new_child2 = pop[i + 1].copy()
            pos = np.random.randint(0, LEN_EN)
            for j in range(pos):
                new_child1[j] = new_child2[j]
            new_pop[i] = new_child1
            new_pop[i + 1] = new_child2
        else:
            new_pop[i] = pop[i]
            new_pop[i + 1] = pop[i+1]
    return new_pop


def mutation(pop):
    new_pop = np.zeros((POP_SIZE, LEN_EN))
    for i in range(POP_SIZE):
        pm = np.random.rand()
        if pm <= P_M:
            pos = np.random.randint(0, LEN_EN)
            if pop[i][pos]:
                pop[i][pos] = 0
            else:
                pop[i][pos] = 1
        new_pop[i] = pop[i]
    return new_pop


while GEN < MAX_GEN:
    ENCODER = encoder(POP)
    FIT = fitness(ENCODER)
    POP = select(FIT, POP)
    POP = crossover(POP)
    POP = mutation(POP)
    GEN = GEN + 1
    MAX_FIT.append(FIT.max())
    MEAN_FIT.append(FIT.mean())

# iteration curve
# plt.plot(MAX_FIT)
# plt.title("Iteration Curve")
# plt.plot(MEAN_FIT)
# plt.show()

# fitness function
x = np.arange(-1, 2, 0.01)
y = fitness(x)
ENCODER = encoder(POP)
Y_ENCODER = fitness(ENCODER)
plt.plot(x, y)
plt.plot(ENCODER, Y_ENCODER)
plt.show()