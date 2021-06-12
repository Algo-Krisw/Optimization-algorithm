# -*— coding: utf-8 -*_
# @Time    : 2021/5/15 下午6:53
# @Author  : Algo
# @Site    :
# @File    : GA_TSP.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt


def distance(c1, c2):
    return np.sqrt(np.sum(np.square(c1 - c2)))


# Hyper parameters
POP_SIZE = 50
P_C = 0.65
P_M = 0.035
MAX_GEN = 5000
GEN = 0
MAX_FIT = []
MEAN_FIT = []
C = np.array([[41, 94], [37, 84], [54, 67], [25, 62], [7, 64], [2, 99],
              [68, 58], [71, 44], [54, 62], [83, 69], [64, 60], [18, 54],
              [22, 60], [83, 46], [91, 38], [25, 38], [24, 42], [58, 69],
              [71, 71], [74, 78], [87, 76], [18, 40], [13, 40], [82, 7],
              [62, 32], [58, 35], [45, 21], [41, 26], [44, 35], [4, 50]])
N_CITY = C.shape[0]
POP = np.array([np.random.permutation(N_CITY) for i in range(POP_SIZE)])
distMatrix = np.array([[distance(C[j], C[i]) for i in range(N_CITY)] for j in range(N_CITY)])
path_dist = []


def fitness(x):
    fit = []
    for i in range(POP_SIZE):
        dist_sum = 0
        temp = x[i].copy()
        for city in range(N_CITY - 1):
            dist_sum += distMatrix[temp[city]][temp[city+1]]
            dist_sum += distMatrix[temp[N_CITY-1]][temp[0]]
        fit.append(dist_sum)
    fit = np.array(fit)
    fit = 10000 / fit
    return fit


def select(fit, pop):
    new_pop = np.zeros((POP_SIZE, N_CITY), dtype=np.int)
    choice_prob = fit / np.sum(fit)
    choice_index = np.arange(0, POP_SIZE)
    index_choice = np.random.choice(choice_index.flatten(), POP_SIZE, p=choice_prob.flatten())
    for i in range(POP_SIZE):
        new_pop[i] = pop[index_choice[i]]
    new_pop[0] = pop[np.argmax(fit)]
    return new_pop


def crossover(pop):
    new_pop = []
    for i in range(POP_SIZE):
        father = pop[i].tolist()
        mother = pop[np.random.randint(0, POP_SIZE)].tolist()
        pc = np.random.random()
        if pc <= P_C:
            pos1 = np.random.randint(0, POP_SIZE - 1)
            pos2 = np.random.randint(pos1, POP_SIZE)
            swap_list = father[pos1: pos2]  # p1-p2为交换部分
            swap_list_2 = mother[pos1: pos2]
            for j in range(N_CITY - pos2):  # 列表右移len - p2 位
                father.insert(0, father.pop())
                mother.insert(0, mother.pop())
            for k in range(len(swap_list)):  # 删除重复部分
                if swap_list_2[k] in father:
                    father.remove(swap_list_2[k])  # 城市1右移后删去和片段2的重复部分
                if swap_list[k] in mother:
                    mother.remove(swap_list[k])  # 城市2右移后删去和片段1的重复部分
            father = father + swap_list_2  # 去重后加入交换片段
            mother = mother + swap_list
            for k in range(len(father) - pos2):  # 左移len - p2 位
                father.insert(len(father), father[0])  # 将第一位复制到最后一位
                father.remove(father[0])  # 删除第一位
                mother.insert(len(mother), mother[0])
                mother.remove(mother[0])
            new_pop.append(father)
            new_pop.append(mother)
        else:
            new_pop.append(father)
            new_pop.append(mother)
    new_pop = np.array(new_pop, dtype=np.int)
    return new_pop


def mutation(pop):
    new_pop = np.zeros((POP_SIZE, N_CITY), dtype=np.int)
    for i in range(0, POP_SIZE):
        pm = np.random.rand()
        if pm <= P_M:
            pos1 = np.random.randint(0, N_CITY-1)
            pos2 = np.random.randint(pos1, N_CITY)
            pop[i][pos1], pop[i][pos2] = pop[i][pos2], pop[i][pos1]
        new_pop[i] = pop[i]
    return new_pop


while GEN < MAX_GEN:
    FIT = fitness(POP)
    POP = select(FIT, POP)
    POP = crossover(POP)
    POP = mutation(POP)
    GEN = GEN + 1
    MAX_FIT.append(10000/FIT.max())
    MEAN_FIT.append(10000/FIT.mean())

path_dist = POP[0]
plt.plot(MAX_FIT)
plt.title("Iteration Curve")
plt.plot(MEAN_FIT)
plt.show()
#
# # routine plot
# x_list = [C[x][0] for x in path_dist]
# y_list = [C[x][1] for x in path_dist]
# x_list.append(C[path_dist[0]][0])
# y_list.append(C[path_dist[0]][1])
# plt.title("Best routine")
# ax = plt.gca()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
# plt.show()
