#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import copy
import numpy as np
from .base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection

from IPython import embed

class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,             #BASE
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        self.func = func_transformer(func)
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # a list of unequal constraint functions with c[i] <= 0
        self.constraint_ueq = list(constraint_ueq)  # a list of equal functions with ceq[i] = 0

        self.Chrom = None              
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

    @abstractmethod
    def chrom2x(self, Chrom):              #BASE
        pass

    def x2y(self):           #BASE
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            #self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
            self.Y = self.Y_raw + 1e2 * penalty_eq + 1e2 * penalty_ueq
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):          #BASE HERE
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)
            global_best_index = np.array(self.generation_best_Y).argmin()
            #Print each epoch
            print("\nEPOCH:", i, "best X", self.X[generation_best_index, :]/(self.X[generation_best_index, :]).sum(), "best Y", self.Y[generation_best_index])
            global_best_index = np.array(self.generation_best_Y).argmin()
            global_best_X = self.generation_best_X[global_best_index]
            global_best_Y = self.func(np.array([global_best_X]))
            print("Global Best X", global_best_X/(global_best_X).sum(), "\nGlobal Best Y", global_best_Y)

        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y

    fit = run


class GA(GeneticAlgorithmBase):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every vaiiables of func
    constraint_eq : list
        equal constraint
    constraint_ueq : list
        unequal constraint
    precision : array_like
        The precision of every vaiiables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, n_dim,               #GA
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7,
                 MAX_TYPE_ALLOWED=4):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array
        self.MAX_TYPE_ALLOWED = int(MAX_TYPE_ALLOWED)

        # Lind is the num of genes of every variable of func（segments）
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, modify: make the ub bigger and add a constraint_ueq
        int_mode = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        # int_mode is an array of True/False. If True, variable is int constraint and need more code to deal with
        for i in range(self.n_dim):
            if int_mode[i]:
                self.constraint_ueq.append(
                    lambda x: x[i] - self.ub[i]
               )
                self.has_constraint = True
                self.ub[i] = self.lb[i] + np.exp2(self.Lind[i]) - 1

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):                       #GA
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):              #GA
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def cut_out_classes(self, X, threshold):
        threshold = threshold.reshape(-1, 1)
        where_droped = np.where(X<threshold)
        where_left = np.where(X>=threshold)
        X[where_droped] = 0
        return X, where_left

    def add_up(self, X, threshold):
        threshold = threshold.reshape(-1, 1)
        X[X==0]=999  #先填充一下999
        where_less = np.where(X<threshold)
        where_ok = np.where(X>=threshold)
        X[X==999]=0
        #less 部分将会被强行补偿成5% 对应的threshold，则其他部分需要分摊这个补偿，以保持比例不变
        #对于这些不足5%的，其"各自行的其他列"分别需要补偿：
        tmp_X = copy.deepcopy(X)
        tmp_X[where_ok]=999
        tmp_X = threshold - tmp_X
        tmp_X[where_ok] = 0
        compensation = tmp_X.sum(axis=1)
        #挑选每一行最大的来补偿，这样才不会赔完了自己反而小于5%
        X[range(X.shape[0]), X.argmax(axis=1)] = X[range(X.shape[0]), X.argmax(axis=1)] - compensation
        X = X + tmp_X  #稍有精度上的差别，不用管了
        return X

    def chrom2x(self, Chrom):            #GA
        cumsum_len_segment = self.Lind.cumsum()
        X_ratio = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X_ratio[:, i] = self.gray2rv(Chrom_temp)
        #X = self.lb + (self.ub - self.lb) * X_ratio

        # By lmw for METAL FUSING:
        #X[-1] = [52., 21.,  0.,  0.,  0.,  7., 20.]
        X = X_ratio

        # 注意Constraints之间的顺序如不合适 则处理起来会很麻烦
        # Constraint 1: 处理输物料最大种类限制，排名范围靠后的先删掉为0，无所为总占比，不用分摊（想被选中的个体需要挤进前几名） 
        sorted_X = copy.deepcopy(X)
        sorted_X.sort() 
        threshold = sorted_X[:, -self.MAX_TYPE_ALLOWED:][:,0]
        X, where_left = self.cut_out_classes(X, threshold)

        #use_dirichlet = True
        use_dirichlet = False
        if not use_dirichlet:
            #如果不用dirichlet,需要手动做重映射，好处是映射关系稳定，坏处是可能损失完备性，或者分布不好，难优化
            X_remap = X**4   #次方后整体向0偏移
            #Constraint 2: Sum=100%
            X_remap_01 = X_remap/X_remap.sum(axis=1).reshape(-1,1)
            #Constraint 3: <5% 分摊补偿
            threshold = X_remap_01.sum(axis=1)*0.05
            X_remap_01 = self.add_up(X_remap_01, threshold)
            #X_remap_01 = np.round(X_remap_01, 2)
            return X_remap_01
        else:
            # Constraint 2: 此处针对于剩余的列（维度暂时缩小），使用狄利克雷函数重映射Sum=100%  （进入排名的个体被分配合理的百分比）
            #分布可以保证每一行和为1, 同时使得分布范围广,但随机生成的狄利克雷函数会使得函数震荡, 即基因数值仅确定排名，具体表征为多少被随机生成所决定，即最终优化得到的结果用料的排名一定是正确的，具体比例随着时间增加趋于最终正确值。
            if len(X[where_left])%self.MAX_TYPE_ALLOWED!=0:
                print("In GA, X % MAX_TYPE_ALLOWED should be 0", "RARE"*1000)
                embed()
                return X
            X_left = X[where_left].reshape(-1, self.MAX_TYPE_ALLOWED)
            dirichlet_map = np.random.dirichlet(np.ones(self.MAX_TYPE_ALLOWED),size=X_left.shape[0]) 
            dirichlet_map.sort(axis=1)   #自己先排序，准备好接受X_left排序
            X_order = X_left.argsort(axis=1)
            ordered_dirichlet_map = np.zeros((self.size_pop, self.MAX_TYPE_ALLOWED))
            for i in range(self.MAX_TYPE_ALLOWED):
                ordered_dirichlet_map[:, i] = dirichlet_map[X_order == i]
            X_left = ordered_dirichlet_map   #X变为了dirichlet map
    
            # Constrain 3：最小需要补成5%，并让该行其他>5%的列来补偿，保持总占比不变 (挤进前几名的个体最后一名如果<5%，则要别的来补偿)
            threshold = X_left.sum(axis=1)*0.05
            X_left = self.add_up(X_left, threshold)
    
            # 最后把维度填充回去，保持维度不变
            X[where_left] = X_left.reshape(-1)
            #X = np.round(X, 2)
            return X

    ranking = ranking.ranking         #GA
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit  #GA
    mutation = mutation.mutation

    def to(self, device):
        '''
        use pytorch to get parallel performance
        '''
        try:
            import torch
            from .operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('pytorch is needed')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):         #GA,   GPU
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)
            X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self


class GA_TSP(GeneticAlgorithmBase):
    """
    Do genetic algorithm to solve the TSP (Travelling Salesman Problem)
    Parameters
    ----------------
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes corresponding to every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    num_points = 8
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print('distance_matrix is: \n', distance_matrix)
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    ```
    Do GA
    ```py
    from sko.GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=8, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.run()
    ```
    """

    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001):           #GA_TSP
        super().__init__(func, n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut)
        self.has_constraint = False
        self.len_chrom = self.n_dim
        self.crtbp()

    def crtbp(self):              #GA_TSP
        # create the population
        tmp = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    def chrom2x(self, Chrom):            #GA_TSP
        return Chrom

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_pmx
    mutation = mutation.mutation_reverse

    def run(self, max_iter=None):    #GA_TSP
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            Chrom_old = self.Chrom.copy()
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # put parent and offspring together and select the best size_pop number of population
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            selected_idx = np.argsort(self.Y)[:self.size_pop]
            self.Chrom = self.Chrom[selected_idx, :]

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.FitV.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y
