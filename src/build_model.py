import argparse
import pandas as pd
import numpy as np
import os,sys,time
import glob
from IPython import embed
import matplotlib.pyplot as plt
import datetime
from scipy.special import comb, perm   #comb(C)<perm(A)
import itertools
from time_counter import calc_time
from sko.DE import DE 
from sko.GA import GA 
import numpy as np
from skopt import gp_minimize
import torch
import copy
from multiprocessing.pool import Pool
from multiprocessing import Manager
from multiprocessing import cpu_count

def schaffer(p):
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    print(x1, x2,   x)
    return x

def get_constraints():   #Constraints are weak.
    #For eq
    string_eq = "100" 
    for i in range(NUM_OF_TYPES_FOR_GA):
        string_eq += " - x[%s]"%i 
    constraint_eq = [
        lambda x: eval(string_eq)  # lambda x: 100 - x[0] - x[1] - x[2] - x[3],   即=0   
        ]

    #For ueq 1
    string_ueq1 = ''
    for i in range(NUM_OF_TYPES_FOR_GA): 
        string_ueq1 += "lambda x: 5 - x[%s],"%i     #lambda x: 5 - x[0], etc....，即x0, x1, x2 ... >5
    string_ueq1 = string_ueq1.strip(',')
    #For ueq 2
    string_ueq2 = ''
    for i in range(NUM_OF_TYPES_FOR_GA):
        string_ueq2 += "x[%s],"%i
    string_ueq2 = string_ueq2.strip(',')
    string_ueq2 = "lambda x: sum(np.array([%s])) - %s"%(string_ueq2, NUM_OF_TYPES_FOR_GA)     #即sum(np.array([x[0], x[1], x[2]....])>0)<=4
    constraint_ueq = list(eval(string_ueq1)) #  + [eval(string_ueq2)]    #两个不等式限制,目前不能加第二个

    constraint_eq = []   #加和小于100, 不注释则清空限制
    constraint_ueq = []   #不注释则清空ueq constraint, 使用上下限lb ub 5-100就可以不注释
    return constraint_eq, constraint_ueq

def wrapper(their_ratio):   #their_ratio是遗传算法给过来的
    their_ratio = their_ratio*(1-sum(INGREDIENT_STORAGE.loc[INGREDIENT_MUST].ratio))
    global this_solution
    this_solution = generate_solution(choices, their_ratio)
    this_solution, element_output = mixing(this_solution)
    _, scores = evaluation(this_solution, element_output)
    return scores

def C(n,k):  
    #import operator
    #return reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))  
    out = comb(n, k)
    return out

def get_storage():
    INGREDIENT_STORAGE = pd.read_csv("../data/0_INGREDIENT_STORAGE.csv", index_col='name')
    which_is_time = np.where(INGREDIENT_STORAGE.columns=='when_comes_in')[0][0]
    #str to datetime
    for row_idx,row in enumerate(INGREDIENT_STORAGE.iterrows()):
        INGREDIENT_STORAGE.iloc[row_idx, which_is_time] = datetime.datetime.strptime(row[1].when_comes_in, "%Y/%m/%d %H:%M")
    assert (INGREDIENT_STORAGE[INGREDIENT_STORAGE.must==0]['ratio'].values==0).all(), "Not must but giving a ratio??"
    assert (INGREDIENT_STORAGE[INGREDIENT_STORAGE.must==1]['ratio'].values!=0).all(), "Lack of ratio for must!"
    assert sum(INGREDIENT_STORAGE[INGREDIENT_STORAGE.must==1]['ratio'].values)<1, "Error giving ratio!"
    return INGREDIENT_STORAGE

def get_ELEMENT_TARGETS():
    ELEMENT_TARGETS_MEAN = pd.read_csv("../data/1_ELEMENT_TARGETS.csv")
    ELEMENT_TARGETS_LOW = ELEMENT_TARGETS_MEAN - ELEMENT_TARGETS_MEAN*0.1
    ELEMENT_TARGETS_HIGH = ELEMENT_TARGETS_MEAN + ELEMENT_TARGETS_MEAN*0.1
    return ELEMENT_TARGETS_LOW, ELEMENT_TARGETS_MEAN, ELEMENT_TARGETS_HIGH

def load_solution():
    SOLUTION = pd.read_csv("../data/2_SOLUTION.csv", index_col='name')
    which_is_percentage = np.where(SOLUTION.columns=='ratio')[0][0]
    #str to percentage
    for row_idx,row in enumerate(SOLUTION.iterrows()):
        SOLUTION.iloc[row_idx, which_is_percentage] = float(row[1]['ratio'].strip("%"))/100
    return SOLUTION

def generate_solution(choices, their_ratio):
    this_solution = INGREDIENT_STORAGE.reindex(index=list(choices), columns=SOLUTION.columns)
    this_solution['ratio'] = their_ratio
    #加上必备行
    this_solution = add_must(this_solution)
    return this_solution

def add_must(this_solution):
    #加上必备的行
    for must_this in INGREDIENT_MUST:
        this_solution.loc[must_this, 'ratio'] = INGREDIENT_STORAGE.loc[must_this, 'ratio']
    return this_solution

def mixing(this_solution):
    if np.round(this_solution['ratio'].sum(), 3) != 1:
        if args.DEBUG:print("***Warning for ratio...", this_solution['ratio'].sum())
        this_solution['ratio'] = this_solution['ratio']/this_solution['ratio'].sum()
    element_output = pd.DataFrame(np.array([0]*len(ELEMENTS)).reshape(1,-1), columns = ELEMENTS)
    for this_type in this_solution.index:
        element_output += this_solution.loc[this_type, 'ratio'] * INGREDIENT_STORAGE.loc[this_type][ELEMENTS]
    first_insufficient_type = this_solution.index[np.where(INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']==min(INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']))[0][0]]
    consumed_amount = (this_solution['ratio']/this_solution.loc[first_insufficient_type, 'ratio'])*INGREDIENT_STORAGE.loc[first_insufficient_type, 'volume_of_storage']
    #after consumed, leftovers are:
    this_solution['consumed_amount'] = consumed_amount
    this_solution['leftover'] = INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'] - consumed_amount
    this_solution['consumed_amount'] = np.round(this_solution['consumed_amount'].astype(float), 2)
    this_solution['leftover'] = np.round(this_solution['leftover'].astype(float), 2)
    return this_solution, element_output

def evaluation(this_solution, element_output):

    #根据混合结果得到Objectives:
    obj_consumed = this_solution['consumed_amount']             #越大越好
    obj_leftover = this_solution['leftover']     #越小越好， 平滑
    obj_leftover_01 =  (this_solution['leftover']/INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']<0.01).sum()     #越大越好, 非平滑, 少于百分之一就算0
    obj_element_diff = abs(ELEMENT_TARGETS_MEAN - element_output)[ELEMENTS_MATTERS]    #越小越好，平滑
    obj_element_01 = list(((ELEMENT_TARGETS_LOW[ELEMENTS_MATTERS] < element_output[ELEMENTS_MATTERS]) & (element_output[ELEMENTS_MATTERS] < ELEMENT_TARGETS_HIGH[ELEMENTS_MATTERS])).loc[0]).count(1)    #越大越好, 非平滑

    #记录
    obj_dict = {}
    obj_dict['obj_consumed'] = obj_consumed
    obj_dict['obj_leftover'] = obj_leftover
    obj_dict['oibj_leftover_01'] = obj_leftover_01
    obj_dict['obj_element_diff'] = obj_element_diff
    obj_dict['obj_element_01'] = obj_element_01

    #Misc
    tmp = list(INGREDIENT_STORAGE['volume_of_storage'])
    tmp.sort()
    volume_normer = sum(tmp[-MAX_TYPE_ALLOWED:])

    #Objectives无量纲化：
    normed_obj_amount = obj_consumed.sum()/volume_normer    #用库存最多的4种总量做标准化  0~1, -->1
    normed_obj_leftover = 1 - obj_leftover.sum()/INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'].sum()   #用这4类数量做标准化  0~1, -->1
    normed_obj_leftover_01 = obj_leftover_01/MAX_TYPE_ALLOWED   #用种类个数做标准化  0~1, -->1
    normed_obj_elements = 1 - 0.01*obj_element_diff.values.sum()/max(1, len(ELEMENTS_MATTERS))    #用需要检查的元素数量做标准化 0~1, -->1
    normed_obj_elements_01 = obj_element_01/len(ELEMENTS_MATTERS)   #用MATTERS的ELEMENT个数做标准化 0~1, -->1

    #记录
    normed_dict = {}
    normed_dict['normed_obj_amount'] = normed_obj_amount
    normed_dict['normed_obj_leftover'] = normed_obj_leftover
    normed_dict['normed_obj_leftover_01'] = normed_obj_leftover_01
    normed_dict['normed_obj_elements'] = normed_obj_elements
    normed_dict['normed_obj_elements_01'] = normed_obj_elements_01

    #Multi-Obj to Single_obj:   #更正认识：平滑的局地极小更多，错误解可能更大, 非平滑的只有在真正正确的时候得到小值
    if args.OBJ == 1:
        objective_function = 5 - args.alpha*normed_obj_amount - args.beta*(normed_obj_leftover+normed_obj_leftover_01) - args.gama*(normed_obj_elements+normed_obj_elements_01)    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    if args.OBJ == 2:
        objective_function = 3 - args.alpha*normed_obj_amount - args.beta*normed_obj_leftover - args.gama*normed_obj_elements    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    if args.OBJ ==3:
        objective_function = 3 - args.alpha*normed_obj_amount - args.beta*normed_obj_leftover_01 - args.gama*normed_obj_elements_01    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    score = objective_function
    if args.DEBUG:print(this_solution, '  ', score)
    return obj_dict, score

def run_opt(struct):
    num = struct
    np.random.seed(num)
    print("Process:", num)
    #Optimization:
    print("Optimization %s, Dimension %s"%(num, NUM_OF_TYPES_FOR_GA))
    constraint_eq, constraint_ueq = get_constraints()
    #wrapper.is_vector=True
    #整数规划，要求某个变量的取值可能个数是2^n，2^n=128, 96+32=128, 则上限为132
    #考虑一步到位,所有物料参与选择,下限为0
    ga = GA(func=wrapper, n_dim=NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*NUM_OF_TYPES_FOR_GA, ub=[100]*NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.0001]*NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_ALLOWED=MAX_TYPE_ALLOWED)
    best_gax, best_gay = ga.run()
    best_ratio = best_gax/best_gax.sum()
    best_solution = copy.deepcopy(this_solution)
    best_solution.loc[INGREDIENT_CHOOSE_FROM, 'ratio'] = best_ratio
    print("Best solution found:\n", best_solution)
    print('Best_ratio:', best_ratio, '\n', 'best_y:', best_gay)

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    return best_gay, best_ratio, best_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--DEBUG', action='store_true', default=False)
    parser.add_argument("-S", '--FROM_SCRATCH', action='store_true', default=False)
    parser.add_argument("-O", '--OBJ', type=int, default=0)
    parser.add_argument("-E", '--epoch', type=int, default=50)
    parser.add_argument("-P", '--pop', type=int, default=100)
    parser.add_argument("-A", '--alpha', type=int, default=1)
    parser.add_argument("-B", '--beta', type=int, default=1)
    parser.add_argument("-G", '--gama', type=int, default=1)
    parser.add_argument("-T", '--threads', type=int, default=int(cpu_count()/2))
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Start....")
    MAX_TYPE_ALLOWED = 4
    ELEMENTS = ['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O']
    ELEMENTS_MATTERS = ['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O']
    #ELEMENTS_MATTERS = ['Cu']

    #获取库存
    INGREDIENT_STORAGE = get_storage()
    INGREDIENT_CHOOSE_FROM = list(INGREDIENT_STORAGE[INGREDIENT_STORAGE.ratio==0].index)
    INGREDIENT_MUST = list(INGREDIENT_STORAGE[INGREDIENT_STORAGE.ratio!=0].index)
    NUM_OF_TYPES_FOR_GA = len(INGREDIENT_CHOOSE_FROM)
    COMBINATIONS = list(itertools.combinations(INGREDIENT_CHOOSE_FROM, NUM_OF_TYPES_FOR_GA))

    #获取元素配比目标
    ELEMENT_TARGETS_LOW, ELEMENT_TARGETS_MEAN, ELEMENT_TARGETS_HIGH = get_ELEMENT_TARGETS()

    #Mannual:
    #1.获取目前的solution
    SOLUTION = load_solution()
    #2.根据目前的SOLUTION混合，得到混合结果
    this_solution = SOLUTION
    this_solution, element_output = mixing(this_solution)
    #3.评判标准
    _, scores = evaluation(this_solution, element_output)
    print('\n', element_output, "\n>>>THEORITICAL BEST SOLUTION YIELDS<<<:", np.round(scores,4))
    #embed()
    #sys.exit()

    #Random search:
    random_search = True
    random_search = False
    if random_search:
        print("\n\nRandom search GA, 1 iters, all pop.")
        best_y = []
        for i in range(args.threads):
            constraint_eq, constraint_ueq = get_constraints()
            choices = list(INGREDIENT_CHOOSE_FROM)
            ga = GA(func=wrapper, n_dim=NUM_OF_TYPES_FOR_GA, size_pop=args.pop*args.epoch, max_iter=1, lb=[0]*NUM_OF_TYPES_FOR_GA, ub=[100]*NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.0001]*NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_ALLOWED=MAX_TYPE_ALLOWED)
            best_gax, best_gay = ga.run()
            best_y.append(best_gay[0])
        best_y = np.array(best_y)
        print("Random search best min:", best_y.mean())
        sys.exit()

    choices = INGREDIENT_CHOOSE_FROM
    print("Choices:", choices)

    blobs = []
    pool = Pool(processes=int(cpu_count()/2))   #这个固定死，效率最高,跟做多少次没关系
    struct_list = range(args.threads)
    rs = pool.map(run_opt, struct_list)              #CORE
    best_ys = np.empty(0)
    best_ratios = np.empty((0,NUM_OF_TYPES_FOR_GA))
    best_solutions = []
    #Re-organize results:
    for r in rs:
        best_y = r[0]
        best_ratio = r[1]
        best_solution = r[2]
        best_ys = np.hstack((best_ys, best_y))
        best_ratios = np.vstack((best_ratios,best_ratio))
        best_solutions.append(best_solution)
    best_ratio = best_ratios[best_ys.argmin()]
    best_solution = best_solutions[best_ys.argmin()]
    print("BEST:", best_solution)
    print(best_ys.min())

    #ga = GA(func=schaffer, n_dim=2, size_pop=6, max_iter=2, lb=[-1, -1], ub=[1, 1], precision=1e-7, prob_mut=0.5)
    #sys.exit()
    #DE没有整数规划API
    #de = DE(func=wrapper, n_dim=NUM_OF_TYPES_FOR_GA, size_pop=100, max_iter=100, lb=[5]*NUM_OF_TYPES_FOR_GA, ub=[100]*NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=1)
    #best_dex, best_dey = de.run() 
    #print('best_dex:', best_dex, '\n', 'best_dey:', best_dey) 


