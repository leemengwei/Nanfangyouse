import tqdm
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
from sko.DE import DE 
from sko.GA import GA 
import numpy as np
#import torch
import copy
from multiprocessing.pool import Pool
from multiprocessing import Manager
from multiprocessing import cpu_count
#NETWORK
from flask import Flask, request, jsonify
from sys import stderr
from flask_cors import cross_origin  #戚总-张驰API
from calc_oxygen import calc_oxygen
app = Flask(__name__)
import random

def get_constraints(args):   #Constraints are weak.
    #For eq
    #string_eq = "100" 
    #for i in range(args.NUM_OF_TYPES_FOR_GA):
    #    string_eq += " - x[%s]"%i 
    #constraint_eq = [
    #    lambda x: eval(string_eq)  # lambda x: 100 - x[0] - x[1] - x[2] - x[3],   即=0   
    #    ]

    ##For ueq 1
    #string_ueq1 = ''
    #for i in range(args.NUM_OF_TYPES_FOR_GA): 
    #    string_ueq1 += "lambda x: 5 - x[%s],"%i     #lambda x: 5 - x[0], etc....，即x0, x1, x2 ... >5
    #string_ueq1 = string_ueq1.strip(',')
    ##For ueq 2
    #string_ueq2 = ''
    #for i in range(args.NUM_OF_TYPES_FOR_GA):
    #    string_ueq2 += "x[%s],"%i
    #string_ueq2 = string_ueq2.strip(',')
    #string_ueq2 = "lambda x: sum(np.array([%s])) - %s"%(string_ueq2, args.NUM_OF_TYPES_FOR_GA)     #即sum(np.array([x[0], x[1], x[2]....])>0)<=4
    #constraint_ueq = list(eval(string_ueq1)) #  + [eval(string_ueq2)]    #两个不等式限制,目前不能加第二个

    constraint_eq = []   #加和小于100, 不注释则清空限制
    constraint_ueq = []   #不注释则清空ueq constraint, 使用上下限lb ub 5-100就可以不注释
    return constraint_eq, constraint_ueq


def adjust_GA_ratio(args, their_ratio):
    their_ratio = their_ratio*(1-sum(args.INGREDIENT_MUST_WITH_RATIO['ratio'].values)) #GA生成的概率sum是100%，但有时可能有“必选且指定比例”项目存在，GA内部仅在5%阈值上是考虑了这个因素的，所以在mix之前调整一下
    return their_ratio

def GAwrapper(their_ratio):   #their_ratio是遗传算法给过来的, GA算法本身的API要求, TODO:their_ratio是一个个给回来的，准备矢量化
    start = time.time()
    #global args
    their_ratio = adjust_GA_ratio(args, their_ratio)
    time_0 = time.time()
    #global this_solution
    this_solution = generate_full_solution(their_ratio)   #加上must with ratio项
    time_1 = time.time()
    this_solution, element_output = mixing(args, this_solution)
    time_2 = time.time()
    global obj_dict
    obj_dict, scores = evaluation(args, this_solution, element_output)
    time_3 = time.time()
    print('0:',100*(time_0-start)/(time.time()-start), '%')
    print('1:',100*(time_1-time_0)/(time.time()-start), '%')
    print('2:',100*(time_2-time_1)/(time.time()-start), '%')
    print('3:',100*(time_3-time_2)/(time.time()-start), '%')
    print('ALl:', time.time()-start)
    return scores

def C(n,k):  
    #import operator
    #return reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))  
    out = comb(n, k)
    return out

def get_storage(for_show=False):
    if for_show:
        ingredient_storage = pd.read_csv("../data/0_INVENTORY_STORAGE_ALL.csv", index_col='name')
    else:
        ingredient_storage = pd.read_csv("../data/1_INVENTORY_STORAGE_CHOOSE_FROM.csv", index_col='name')
    for this_element in args.ELEMENTS:
        ingredient_storage[this_element] = np.round(ingredient_storage[this_element], 6)
    #which_is_time = np.where(ingredient_storage.columns=='when_comes_in')[0][0]
    #str to datetime
    #for row_idx,row in enumerate(ingredient_storage.iterrows()):
    #    ingredient_storage.iloc[row_idx, which_is_time] = datetime.datetime.strptime(row[1].when_comes_in, "%Y/%m/%d %H:%M")
    return ingredient_storage

def get_elements_boundary(args):
    args.ELEMENT_TARGETS_LOW = args.ELEMENT_TARGETS_MEAN - args.ELEMENT_TARGETS_MEAN*0.01
    args.ELEMENT_TARGETS_HIGH = args.ELEMENT_TARGETS_MEAN + args.ELEMENT_TARGETS_MEAN*0.01
    return args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_HIGH

def load_solution():
    SOLUTION = pd.read_csv("../data/3_SOLUTION.csv", index_col='name')
    which_is_percentage = np.where(SOLUTION.columns=='ratio')[0][0]
    #str to percentage
    for row_idx,row in enumerate(SOLUTION.iterrows()):
        SOLUTION.iloc[row_idx, which_is_percentage] = float(row[1]['ratio'].strip("%"))/100
    return SOLUTION

def generate_full_solution(their_ratio):
    part_solution = args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN
    part_solution['ratio'] = their_ratio
    #加上必备行
    full_solution = add_solution_with_must_with_ratio_only(part_solution)  #GA出来已有‘仅必选’&‘必清空’
    return full_solution

#def add_solution_with_must_ratio_and_must_clean(part_solution):    #决定不走这条路了，还是把‘仅必选、必选且用光’两个都传给GA，之后对后者再惩罚吧。
#    #a) GA出来已经包含了’仅必备’，这里首先考虑加上‘必备且有百分比’的行：
#    for must_this_with_ratio in args.INGREDIENT_MUST_WITH_RATIO:
#        part_solution.loc[must_this_with_ratio, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_this_with_ratio, 'ratio']
#    #b) 目前仅剩下‘必备清仓’未考虑了，加上"必备清仓"，这里稍复杂，需要在本处根据GA输出各项目的比例，动态给出清仓项比例。
#    tmp_consumption = get_consumed_amounts(part_solution)
#    consumption_with_must_clean = sum(tmp_consumption) + sum(args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_CLEAN,'volume_of_storage'])
#    for must_clean in args.INGREDIENT_MUST_CLEAN:
#        part_solution.loc[must_clean, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_clean, 'volume_of_storage']/consumption_with_must_clean
#    #c) 再根据消耗量反算其它项目，但显而易见这会影响到其他原为5%的项目，导致他们的比例更低：  #感觉不好解决, turn back
#    part_solution.loc[tmp_consumption.index, 'ratio'] = tmp_consumption/consumption_with_must_clean
#    full_solution = part_solution
#    return full_solution

def add_solution_with_must_with_ratio_only(part_solution):  #目前走这条路，注意需要确认已经把‘仅必选、必选且用光’两个都传给GA了，则此处只补充‘必备且有百分比’的项目。
    full_solution = pd.concat([part_solution, args.INGREDIENT_MUST_WITH_RATIO])
    #for must_this_with_ratio in args.INGREDIENT_MUST_WITH_RATIO.index:
    #    part_solution.loc[must_this_with_ratio, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_this_with_ratio, 'ratio'] 
    #full_solution = part_solution
    np.concatenate([part_solution.values, args.INGREDIENT_MUST_WITH_RATIO.values])
    return full_solution

def get_consumed_amounts(this_solution):
    first_insufficient_type = this_solution.index[np.where(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']==min(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']))[0][0]]
    consumed_amounts = (this_solution['ratio']/this_solution.loc[first_insufficient_type, 'ratio'])*args.INGREDIENT_STORAGE.loc[first_insufficient_type, 'volume_of_storage']
    return consumed_amounts

def mixing(args, this_solution):
    if args.MAX_TYPE_TO_SEARCH != 0:
        if not this_solution.loc[this_solution.index[0], 'ratio'] < np.inf:    #当待选1个，precision=1,GA内部会出现唯一一个取为0情况导致出现nan，修订为1-ratio_taken.
            print("*"*88)
            this_solution.loc[this_solution.index[0], 'ratio'] = 1 - sum(args.INGREDIENT_MUST_WITH_RATIO['ratio'])
    if np.round(this_solution['ratio'].sum(), 3) != 1:
        if args.DEBUG:print("***Warning for ratio...", this_solution['ratio'].sum())
        this_solution['ratio'] = this_solution['ratio']/this_solution['ratio'].sum()
    element_output = pd.DataFrame(np.array([0]*len(args.ELEMENTS)).reshape(1,-1), columns = args.ELEMENTS)
    for this_type in this_solution.index:
        element_output += this_solution.loc[this_type, 'ratio'] * args.INGREDIENT_STORAGE.loc[this_type][args.ELEMENTS]
    #after consumed, leftovers are:
    this_solution['consumed_amounts'] = get_consumed_amounts(this_solution)
    this_solution['leftover'] = args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'] - this_solution['consumed_amounts']
    this_solution['productionTime'] = np.round(this_solution['consumed_amounts']/(this_solution['ratio']*args.Flow), 2)
    this_solution['productionTime'] = np.round(this_solution['volume_of_storage']/(this_solution['ratio']*args.Flow), 2)
    return this_solution, element_output

def evaluation(args, this_solution, element_output):
    evaluate_on = (this_solution['ratio'] != 0).index
    #根据混合结果得到Objectives:
    obj_consumed = this_solution.loc[evaluate_on, 'consumed_amounts']             #越大越好
    obj_leftover = this_solution.loc[evaluate_on, 'leftover'] #越小越好， 平滑
    obj_leftover_01 =  (this_solution.loc[evaluate_on, 'leftover']/args.INGREDIENT_STORAGE.loc[evaluate_on, 'volume_of_storage']<0.01).sum()     #越大越好, 非平滑, 少于百分之一就算0
    obj_element_diff = abs(args.ELEMENT_TARGETS_MEAN - element_output)[args.ELEMENT_MATTERS]    #越小越好，平滑
    obj_element_01 = list(((args.ELEMENT_TARGETS_LOW[args.ELEMENT_MATTERS] < element_output[args.ELEMENT_MATTERS]) & (element_output[args.ELEMENT_MATTERS] < args.ELEMENT_TARGETS_HIGH[args.ELEMENT_MATTERS])).loc[0]).count(1)    #越大越好, 非平滑

    #记录
    obj_dict = {}
    obj_dict['obj_consumed'] = obj_consumed
    obj_dict['obj_leftover'] = obj_leftover
    obj_dict['obj_leftover_01'] = obj_leftover_01
    obj_dict['obj_element_diff'] = obj_element_diff
    obj_dict['obj_element_01'] = obj_element_01

    #Misc
    tmp = list(args.INGREDIENT_STORAGE.loc[evaluate_on, 'volume_of_storage'])
    tmp.sort()
    volume_normer = sum(tmp[-args.MAX_TYPE_TO_SEARCH:])
    #leftover_normer = sum(args.INGREDIENT_STORAGE.loc[evaluate_on, 'volume_of_storage'][(this_solution.loc[evaluate_on, 'consumed_amounts']!=0).values])
    leftover_normer = args.INGREDIENT_STORAGE.loc[evaluate_on, 'volume_of_storage'].sum()

    #Objectives无量纲化：
    normed_obj_amount = obj_consumed.sum()/volume_normer    #用库存最多N种总量做标准化(不包含必选项目)  0~1, -->1
    normed_obj_leftover = 1 - obj_leftover.sum()/leftover_normer   #用所有可选择的类数量做标准化  0~1, -->1
    normed_obj_leftover_01 = obj_leftover_01/max(args.MAX_TYPE_TO_SEARCH,1)   #用种类个数做标准化  0~1, -->1
    normed_obj_elements = 1 - 0.01*(obj_element_diff*args.ELEMENT_PRIORITIES_SCORE).values.sum()    #用需要检查的元素数量做标准化 0~1, -->1
    normed_obj_elements_01 = obj_element_01/len(args.ELEMENT_MATTERS)   #用MATTERS的ELEMENT个数做标准化 0~1, -->1
 
    #记录
    #normed_dict = {}
    global normed_dict
    normed_dict['normed_obj_amount'].append(normed_obj_amount)
    normed_dict['normed_obj_leftover'].append(normed_obj_leftover)
    normed_dict['normed_obj_leftover_01'].append(normed_obj_leftover_01)
    normed_dict['normed_obj_elements'].append(normed_obj_elements)
    normed_dict['normed_obj_elements_01'].append(normed_obj_elements_01)
    #print(normed_dict, "ele_diff:", obj_element_diff, 'priorities', args.ELEMENT_PRIORITIES_SCORE)

    #Multi-Obj to Single_obj:   #更正认识：平滑的局地极小更多，错误解可能更大, 非平滑的只有在真正正确的时候得到小值
    if args.OBJ == 1:    #平滑+非平滑
        objective_function = (args.alpha+2*args.beta+2*args.gama) - args.alpha*normed_obj_amount - args.beta*(normed_obj_leftover+normed_obj_leftover_01) - args.gama*(normed_obj_elements+normed_obj_elements_01)    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    if args.OBJ == 2:    #完全平滑
        objective_function = (args.alpha+args.beta+args.gama) - args.alpha*normed_obj_amount - args.beta*normed_obj_leftover - args.gama*normed_obj_elements    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    if args.OBJ ==3:    #非平滑
        objective_function = (args.alpha+args.beta+args.gama) - args.alpha*normed_obj_amount - args.beta*normed_obj_leftover_01 - args.gama*normed_obj_elements_01    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    score = objective_function
    if args.DEBUG:print(this_solution, '  ', score)
    return obj_dict, score

def run_opt_map(struct):   #map需要，多线程调用GA
    num = struct[0]
    args = struct[1]
    for i in range(100):
        seed = int(str(time.time()).split('.')[-1])
        time.sleep(seed/1e9)
        np.random.seed(seed)
    print("Process:", num, seed)
    print("Optimization %s, Dimension %s"%(num, args.NUM_OF_TYPES_FOR_GA))
    constraint_eq, constraint_ueq = get_constraints(args)
    #GAwrapper.is_vector=True
    #整数规划，要求某个变量的取值可能个数是2^n，2^n=128, 96+32=128, 则上限为132
    #考虑一步到位,所有物料参与选择,下限为0
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.01]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.003, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=sum(args.INGREDIENT_MUST_WITH_RATIO['ratio']), columns_just_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
    best_gax, best_gay = ga.run()
    best_ratio = best_gax/best_gax.sum()
    best_solution = generate_full_solution(best_ratio)
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index, 'ratio'] = best_ratio

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    return best_gay, best_ratio, best_solution

def run_rand(args):
    print("\n\nRandom search GA, 1 iters, all pop.")
    best_ys = []
    for i in range(args.threads):  #多线程跑几次这里就跑几次
        constraint_eq, constraint_ueq = get_constraints(args)
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop*args.epoch, max_iter=1, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.01]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.003, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=sum(args.INGREDIENT_MUST_WITH_RATIO['ratio']), columns_just_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
        best_gax, best_gay = ga.run()
        best_ys.append(best_gay[0])
    best_ys = np.array(best_ys)
    print("***Random search best mean:", best_ys.mean(), best_ys.min())

def run_opt(args):
    blobs = []
    pool = Pool(processes=int(cpu_count()/2))   #这个固定死，效率最高,跟做多少次没关系
    struct_list = []
    for i in range(args.threads):  #做threads次
        struct_list.append([i, args])
    rs = pool.map(run_opt_map, struct_list) #CORE
    pool.close()
    pool.join()
    best_ys = np.empty(0)
    best_ratios = np.empty((0,args.NUM_OF_TYPES_FOR_GA))
    best_solutions = []
    #Re-organize results:
    for r in rs:
        best_y = r[0]
        best_ratio = r[1]
        best_solution = r[2]
        best_ys = np.hstack((best_ys, best_y))
        best_ratios = np.vstack((best_ratios,best_ratio))
        best_solutions.append(best_solution)
    best_adjust_GA_ratio = adjust_GA_ratio(args, best_ratios[best_ys.argmin()])   #记得调整一下
    best_solution = best_solutions[best_ys.argmin()]
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index, 'ratio'] = best_adjust_GA_ratio
    best_y = best_ys[best_ys.argmin()]
    print("***BEST:", best_solution)
    print(best_ys.min())
    _, element_output = mixing(args, best_solution)
    return best_adjust_GA_ratio, best_y, best_solution, element_output

#For server~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pd_to_res(storage):
    res_data = [] 
    key_copying = {'ratio':'calculatePercentage', 'leftover':'inventoryBalance', 'volume_of_storage':'inventory'}
    for i in storage.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
            #if key_attr == 'required' or key_attr == 'clean' or key_attr == 'cohesion':   #这个key特殊处理一下true false
            #    this_dict[key_attr] = True if i[1][this_attr] == 1 else False
            #else:  #貌似不特殊处理才对,特殊处理会影响可编辑性。
            this_dict[key_attr] = i[1][this_attr]
            if key_attr in key_copying.keys():
                key_attr = key_copying[key_attr]
                this_dict[key_attr] = i[1][this_attr]    #copy to another one
        res_data.append(this_dict) 
    return res_data 

def compelete_basic_args(args):
    #获取库存 for 计算
    if not args.ON_SERVER:
        args.INGREDIENT_STORAGE = get_storage()
    else:
        pass
    args.INGREDIENT_MUST_WITH_RATIO = args.INGREDIENT_STORAGE.loc[list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio!=0].index))]   #必选比例以定
    args.INGREDIENT_MUST_CLEAN = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean!=0].index)]  #必选且必须清空该料
    args.INGREDIENT_JUST_MUST = args.INGREDIENT_STORAGE.loc[list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio==0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean==0].index))]  #必选但不指定比例
    args.INGREDIENT_CHOOSE_FROM = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_STORAGE[(args.INGREDIENT_STORAGE.required+args.INGREDIENT_STORAGE.clean)==0].index)]
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_CHOOSE_FROM.index) + list(args.INGREDIENT_JUST_MUST.index)]
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_WITH_RATIO = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_CHOOSE_FROM.index) + list(args.INGREDIENT_JUST_MUST.index) + list(args.INGREDIENT_MUST_WITH_RATIO.index)]
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_CHOOSE_FROM.index) + list(args.INGREDIENT_JUST_MUST.index) + list(args.INGREDIENT_MUST_CLEAN.index)]
    args.JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_JUST_MUST.index) + list(args.INGREDIENT_MUST_CLEAN.index)]
    #整理一下顺序, 要给GA准备辅助的位置，来简化must clean和just must两个项目
    #order = list(args.INGREDIENT_STORAGE.index).index
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST = args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST.reindex(args.INGREDIENT_STORAGE.index).dropna()
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_WITH_RATIO = args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_WITH_RATIO.reindex(args.INGREDIENT_STORAGE.index).dropna()
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.reindex(args.INGREDIENT_STORAGE.index).dropna()
    args.NUM_OF_TYPES_FOR_GA = len(args.INGREDIENT_CHOOSE_FROM) + len(args.INGREDIENT_JUST_MUST) + len(args.INGREDIENT_MUST_CLEAN)
    args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_HIGH = get_elements_boundary(args)
    #对于必清的项目，计算其相互的比例倍数，准备通过dimension reducer给GA算法
    if len(args.INGREDIENT_MUST_CLEAN)>0:
        dimension_reducer = args.INGREDIENT_MUST_CLEAN['volume_of_storage']/min(args.INGREDIENT_MUST_CLEAN['volume_of_storage'])
    else:
        dimension_reducer = pd.DataFrame([])
    args.DIMENSION_REDUCER_DICT = {} 
    for i in dimension_reducer.index: 
        args.DIMENSION_REDUCER_DICT[list(args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index).index(i)] = dimension_reducer[i]
    #另外需要给ga准备just must(和must clean)的col index
    args.JUST_MUST_AND_MUST_CLEAN_COLUMNS = []
    for i in args.JUST_MUST_AND_MUST_CLEAN.index:
        args.JUST_MUST_AND_MUST_CLEAN_COLUMNS.append(list(args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index).index(i))
    return args

def oxygen_ok(oxygenMaterialRatio_1, oxygenMaterialRatio_2, tmp_oxygenMaterialRatio):
    if oxygenMaterialRatio_2 > oxygenMaterialRatio_1:   #第二单的氧料比更大
        if tmp_oxygenMaterialRatio>oxygenMaterialRatio_1 and tmp_oxygenMaterialRatio<oxygenMaterialRatio_2:
            return True
        else:
            return False
    else:
        if tmp_oxygenMaterialRatio<oxygenMaterialRatio_1 and tmp_oxygenMaterialRatio>oxygenMaterialRatio_2:
            return True
        else:
            return False

@app.route('/api/getFormula', methods=['GET'])
@cross_origin()
def getFormula():   #NOTE: temporary  # API 0-2
    #随便搜索两个:
    global args
    if os.path.exists('../data/solution_1.csv') and os.path.exists('../data/solution_2.csv'):
        solution_1 = pd.read_csv("../data/solution_1.csv", index_col=0)
        solution_2 = pd.read_csv("../data/solution_2.csv", index_col=0)
        #embed()
        _, element_output_1 = mixing(args, copy.deepcopy(solution_1))
        _, element_output_2 = mixing(args, copy.deepcopy(solution_2))
        print("Solution read in")
    else:
        solution_1 = args.INGREDIENT_STORAGE.loc[random.sample(list(args.INGREDIENT_STORAGE.index), 5)]
        solution_2 = args.INGREDIENT_STORAGE.loc[random.sample(list(args.INGREDIENT_STORAGE.index), 5)]
        solution_1['ratio'] = np.random.dirichlet(range(1,len(solution_1.index)+1))
        solution_2['ratio'] = np.random.dirichlet(range(1,len(solution_2.index)+1))
        solution_1['cohesion'] = 0    #NOTE , temporary
        solution_2['cohesion'] = 0    #NOTE , temporary
        solution_1 = web_ratio_int(solution_1)
        solution_2 = web_ratio_int(solution_2)
        solution_1, element_output_1 = mixing(args, solution_1)
        solution_2, element_output_2 = mixing(args, solution_2)
        solution_1 = web_consumption_int(solution_1)
        solution_2 = web_consumption_int(solution_2)
        solution_1.to_csv("../data/solution_1.csv")
        solution_2.to_csv("../data/solution_2.csv")
        print("Randomly generated")
 
    solution_1['formula'] = '1'
    solution_2['formula'] = '2'
    oxygenMaterialRatio_1, Matte_T1, Slag_T1, Wind_Flux1, SiO2_T1 = calc_oxygen(args, element_output_1)
    oxygenMaterialRatio_2, Matte_T2, Slag_T2, Wind_Flux2, SiO2_T2 = calc_oxygen(args, element_output_2)
    embed()

    res_data1 = pd_to_res(solution_1)
    res_data2 = pd_to_res(solution_2)
    res_data = {
        "list": 
            [res_data1, res_data2],
        'materialList':
            [
              {'formula':solution_1['formula'][0],
              'elementsList':compute_element_overview(args, solution_1)},
              {'formula':solution_2['formula'][0],
              'elementsList':compute_element_overview(args, solution_2)}
            ],
        "oxygenMaterialRatio":
            {
              "formula1": round(oxygenMaterialRatio_1, 2),
              "formula2": round(oxygenMaterialRatio_2, 2),
              "formula*": '计算后下方显示',
            },
         }
    return jsonify(res_data)


@app.route('/api/quick_recommend', methods=['POST', 'GET'])
@cross_origin()
def quick_recommend():   #API 3  
    req_data = request.get_json()
    global args
    solution_1 = []
    solution_2 = []
    for i in req_data['list'][0]:
        solution_1.append(i)
    for i in req_data['list'][1]:
        solution_2.append(i)
    solution_1 = req_to_pd(solution_1)
    solution_2 = req_to_pd(solution_2)
    #NOTE:实际上需要传订单1（被衔接）当时的库存状态。
    solution_1['leftover'] = solution_1['inventoryBalance']   #inventoryBalance是通过页面修改回传的
    solution_2['leftover'] = solution_2['inventoryBalance']   #inventoryBalance是配方2生产后理论剩余
    solution_1['volume_of_storage'] = copy.deepcopy(solution_1['leftover'])
    solution_2['volume_of_storage'] = copy.deepcopy(solution_2['leftover'])
    solution_1['inventory'] = solution_1['volume_of_storage']
    solution_2['inventory'] = solution_2['volume_of_storage']
    args.INGREDIENT_STORAGE.loc[solution_2.index] = solution_2
    args.INGREDIENT_STORAGE.loc[solution_1.index] = solution_1   
    def get_compose_solution_from_to(solution_2, solution_1):   #Compose from 2 to 1
        concat_oxygen = 99999
        concat_solution = pd.DataFrame([])
        concat_oxygen = np.array([])
        #oxygenMaterialRatio_1, Matte_T1, Slag_T1, Wind_Flux1, SiO2_T1 = calc_oxygen(args, mixing(args, solution_1)[1])
        #oxygenMaterialRatio_2, Matte_T2, Slag_T2, Wind_Flux2, SiO2_T2 = calc_oxygen(args, mixing(args, solution_2)[1])
        oxygenMaterialRatio_1 = float(req_data['oxygenMaterialRatio']['formula1'])
        oxygenMaterialRatio_2 = float(req_data['oxygenMaterialRatio']['formula2'])

        #穷举计算所有compose solution
        oxygenMaterialRatios = []
        compose_solutions = []
        compose_leftovers_sorter = []    #这个sorter仅统计未消耗光的其他项的数量，越小越好
        solution_1_short_types = solution_1[solution_1['cohesion'] == 1].index  #让用户选择需要衔接哪一个吧
        solution_2_short_types = solution_2[solution_2['leftover'] == 0].index
        solution_2_types_avaliable = list(set(solution_2.index) - set(solution_2_short_types) - set(solution_1.index))  #NOTE: 1、2相同的项不在这个变量中考虑，下面会补充搜索‘不添加任何项’来考虑。
        if not (len(solution_1_short_types)<=len(solution_2_types_avaliable)):
            status = "Error, 除去相同项后(相同项目不应存在衔接需求)，2号配料单其他可用的项目（生产后不为0）已经不够"
            print(status)
            print(solution_1)
            print(solution_2)
            return solution_1, oxygenMaterialRatio_1, status
        combinations_more_to_less = list(itertools.combinations(list(solution_2.loc[solution_2_types_avaliable].sort_values('leftover').index[::-1]), len(solution_1_short_types)))
        combinations_more_to_less.insert(0, '')   #添加一个空项目进来，即‘不混入任何配方2’。
        #把新的2混到旧的1中
        print(combinations_more_to_less)
        for i in combinations_more_to_less:
            tmp_solution_1 = copy.deepcopy(solution_1.drop(solution_1_short_types))   #每个旧的耗尽项都空出来
            #tmp_solution_1 = pd.concat([tmp_solution_1, solution_2.loc[list(i)]])  #接上新加的项
            print("Searching... +", i, tmp_solution_1.index)
            #随机搜索配比组合：
            for _ in tqdm.tqdm(range(500)):
                tmp_solution_1['ratio'] = np.random.dirichlet(range(1, len(tmp_solution_1)+1))
                #计算每一种组合的情况
                tmp_solution_1, tmp_element_output = mixing(args, tmp_solution_1)   #mix之后就有新的消耗列了，然后在计算混入项的理论剩余（下面)
                tmp_oxygenMaterialRatio, tmp_Matte_T, tmp_Slag_T, tmp_Wind_Flux, tmp_SiO2_T = calc_oxygen(args, tmp_element_output)
                #对于那些氧料比需要满足要求的，（最终物料存量也会满足要求，因为mix时用的是两个配方的剩余量）：
                if oxygen_ok(oxygenMaterialRatio_1, oxygenMaterialRatio_2, tmp_oxygenMaterialRatio): # and (tmp_solution_1['ratio']>=0.05).all():
                    compose_solutions.append(copy.deepcopy(tmp_solution_1))
                    oxygenMaterialRatios.append(tmp_oxygenMaterialRatio)
                    compose_leftovers_sorter.append(tmp_solution_1['leftover'].sum())
        #查找结束后看是否存在可行解，取出其中旧单子所有物料剩余最小的情况作为解：
        if len(compose_solutions)>=1:
            concat_solution = compose_solutions[np.array(compose_leftovers_sorter).argmin()]
            concat_oxygen = oxygenMaterialRatios[np.array(compose_leftovers_sorter).argmin()]
            add_ons = '（不填加任何配方2项目，仅调整比例即可满足继续生产需求）' if len(concat_solution)!=len(solution_1) else ''
            status = "Okay, 搜索完毕，在所有可行解组合中找到的物料剩余较小情况解已给出，搜索粒度较为粗糙，可考虑手动微调比例" + add_ons
            if sum(concat_solution['consumed_amounts'])==0:
                status = 'Warning, 目前总消耗为0, 请检查并衔接配方1中的0库存量物料'
        else:
            status = "Error, 衔接搜索结束，暂无法满足氧料比要求，可尝试再试一次或人工衔接"
            print(status)
            print(solution_1, oxygenMaterialRatio_1)
            print(solution_2, oxygenMaterialRatio_2)
            return solution_1, oxygenMaterialRatio_1, status
        return concat_solution, concat_oxygen, status
    #衔接，单项混入（单新入旧），穷举所有情况及其得分
    #NOTE: 三个简化：1、剩下的N项按照其各自剩余量确定其比例（原则上为了尽可能同时用完）；2、必加一新项（原则上为了逐步衔接新的订单）；3、每次衔接不考虑“衔接之后再衔接”，即当前满足了氧料比落在之间开始生产，至于二次此次衔接是否会使得下次“氧料比区间”求解困难，不再过多考虑（实际上呼应了1,我们简化认为一次衔接后剩余的都是不需要处理的小量）；
    #算法：从新单最大库存的物料开始搜索，占比5～30%，配合旧N项，氧料比落在两者之间即退出！若穷尽后无法满足，则不考虑数值约束，直接选择新料中Fe S含量最近的物料以原比例代替，并给出提示。
    concat_solution, concat_oxygenMaterialRatio, status = get_compose_solution_from_to(solution_2, solution_1)
    concat_solution = web_ratio_int(concat_solution)
    concat_solution, concat_element_output = mixing(args, concat_solution)
    concat_solution = web_consumption_int(concat_solution)
    concat_oxygenMaterialRatio, concat_Matte_T, concat_Slag_T, concat_Wind_Flux, concat_SiO2_T = calc_oxygen(args, concat_element_output)  
    #pandas to req_data
    res_element = pd_to_res(concat_element_output)[0]
    res_data = pd_to_res(concat_solution)
    new_res_element = [] 
    for key in res_element.keys(): 
        if key=='name':continue 
        new_res_element.append({'name':key, 'percentage':np.round(res_element[key], 2)})
    S_vs_Cu = res_element['S']/res_element['Cu']

    res_data = {
        "list": 
            res_data,
        "calculateParameter":
        {
            "oxygenMaterialRatio": round(concat_oxygenMaterialRatio, 2),
            "": round(concat_oxygenMaterialRatio, 2),
            "totalConsumedAmount": round(sum(concat_solution['consumed_amounts']), 2),
            "totalLeftOver": round(sum(concat_solution['leftover']), 2),
            "best_y": round(0.000, 2),
            "paFlow": round(concat_Wind_Flux, 2),
            "SCuRatio": round(S_vs_Cu, 2),
            "totalMatte": round(concat_Matte_T, 2),
            "totalSlag": round(concat_Slag_T, 2),
            "totalQuartz": round(concat_SiO2_T, 2),
        },
        "elementsMixtureList": 
            new_res_element,
        "recommended": str(status)
    }
    return jsonify(res_data)

@app.route('/api/quick_update2', methods=['POST', 'GET'])
@cross_origin()
def quick_update2():
    res_data = quick_update(by_update_2=True).json
    res_data['recommended'] = '手动调整返回'
    return jsonify(res_data)

@app.route('/api/quick_update', methods=['POST', 'GET'])
@cross_origin()
def quick_update(by_update_2=False):   #API 2  NOTE: 这个页面调整项涉及到的物料，其所使用的库存应该是考虑上张单子的消耗，但本身这个调整是在配出配料单之后的动作，所以这应该是自然发生的，应该不需要和周工沟通
    req_data = request.get_json()
    global args
    web_solution = req_to_pd(req_data['list'])
    old_ratio = copy.deepcopy(web_solution['ratio'])
    #如果网页回传了adjustRatio，则接下来mix所用的ratio响应调整。
    try:
        for i in web_solution.iterrows():
            if web_solution.loc[i[0], 'adjustRatio'] >= 0:
                web_solution.loc[i[0], 'ratio'] = web_solution.loc[i[0], 'adjustRatio']
                web_solution.loc[i[0], 'calculatePercentage'] = web_solution.loc[i[0], 'adjustRatio']
            else:
                web_solution.loc[i[0], 'adjustRatio'] = web_solution.loc[i[0], 'ratio']
    except:
        for i in web_solution.iterrows():
            web_solution.loc[i[0], 'adjustRatio'] = web_solution.loc[i[0], 'ratio']
    web_solution = web_ratio_int(web_solution)
    adjust_solution, element_output = mixing(args, web_solution)
    adjust_solution = web_consumption_int(adjust_solution)
    _, _y_ = evaluation(args, adjust_solution, element_output)
    
    #计算氧料比update：
    oxygenMaterialRatio, Matte_T, Slag_T, Wind_Flux, SiO2_T = calc_oxygen(args, element_output)
    #pandas to req_data
    res_element = pd_to_res(element_output)[0]
    res_data = pd_to_res(adjust_solution)
    new_res_element = [] 
    for key in res_element.keys(): 
        if key=='name':continue 
        new_res_element.append({'name':key, 'percentage':np.round(res_element[key], 2)})
    S_vs_Cu = res_element['S']/res_element['Cu']

    res_data = {
        "list": 
            res_data,
        "calculateParameter":
        {
            "oxygenMaterialRatio": round(oxygenMaterialRatio, 2),
            "totalConsumedAmount": round(sum(adjust_solution['consumed_amounts']), 2),
            "totalLeftOver": round(sum(adjust_solution['leftover']), 2),
            "best_y": round(_y_, 2),
            "paFlow": round(Wind_Flux, 2),
            "SCuRatio": round(S_vs_Cu, 2),
            "totalMatte": round(Matte_T, 2),
            "totalSlag": round(Slag_T, 2),
            "totalQuartz": round(SiO2_T, 2),
        },
        "elementsMixtureList": 
            new_res_element
    }
    if by_update_2:
        return res_data
    return jsonify(res_data)

def req_to_pd(req_list):
    pd_data = pd.DataFrame()
    for i in req_list: 
        pd_data = pd_data.append(pd.DataFrame(data=i, index=[i['name']]))
    try:
        pd_data.clean = pd_data.clean+0
    except:
        pass
    try:
        pd_data.required = pd_data.required+0  #这样就把网页传回来的true false改成 01 了
    except:
        pass
    try:
        pd_data['ratio'] = pd_data['calculatePercentage']  #这样就把网页传回来的calcPrecent 改成了ratio
    except:
        pass
    return pd_data

#Web Show INT:
def web_ratio_int(best_solution):
    try:
        interger_ratio = np.round(best_solution.ratio, 2)
        need_to_add = int(np.round((1-interger_ratio.sum())*100))
        if need_to_add!=0:
            #各项余数tmp
            drifts = best_solution.ratio - interger_ratio
            drifts_ascending = drifts.sort_values()
            if need_to_add>0: #不到100%需要补充 
                for i in range(abs(need_to_add)): 
                    interger_ratio[drifts_ascending.index[-(i+1)]] += 0.01
                    print("Adding ", drifts_ascending.index[-(i+1)])
            else:  # need_to_add<0:
                for i in range(abs(need_to_add)): 
                    interger_ratio[drifts_ascending.index[i]] -= 0.01
                    print("Cutting ", drifts_ascending.index[i])
        best_solution.ratio = np.round(interger_ratio, 6)  #Web display bug
    except:
        print("Ratio error!! pass")
    return best_solution

def web_consumption_int(best_solution):
    best_solution['consumed_amounts'] = np.clip(np.round(get_consumed_amounts(best_solution)).astype(int), 0, best_solution['volume_of_storage'])
    best_solution['leftover'] = best_solution['volume_of_storage'] - best_solution['consumed_amounts']
    #for index,content in best_solution.iterrows():   #如果页面上想展示detial：
    #    best_solution.loc[index, 'ratio'] = str(best_solution.loc[index, 'ratio'])+" (%s%%)"%np.round(raw_ratio.loc[index]*100,2)
    return best_solution

def compute_element_overview(args, storage):
    new_res_element = []
    for this_element in args.ELEMENTS:
        new_res_element.append({'name': this_element, 'percentage': np.round(sum(storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), 'volume_of_storage']*storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), this_element]) / sum(storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), 'volume_of_storage']), 2)})
    return new_res_element

@app.route('/api/calculate', methods=['POST', 'GET'])
@cross_origin()
def calculate():    #API 1,
    req_data = request.get_json()
    global args

    #req_data to pd:
    pd_data = req_to_pd(req_data['list'])
    args.INGREDIENT_STORAGE = pd_data   #NOTE 接收的配料基础数据是当前的库存，周工必须这么传给我

    #Web-set parameters
    args.Fe_vs_SiO2 = req_data['presetParameter']['FeSiO2Ratio']
    args.Flow = req_data['presetParameter']['consumedAmount']
    args.Matte_Fe_Percentage = req_data['presetParameter']['matteFePercentage']
    args.Matte_S_Percentage = req_data['presetParameter']['matteSPercentage']
    args.Matte_Cu_Percentage = req_data['presetParameter']['matteTargetGradePercentage']
    args.MAX_TYPE_TO_SEARCH = req_data['presetParameter']['maxType']
    args.OXYGEN_PEER_COAL = req_data['presetParameter']['oxygenPeaCoalRatio']
    args.OXYGEN_CONCENTRATION = req_data['presetParameter']['oxygenConcentration']
    args.COAL_T = req_data['presetParameter']['peaCoal']
    args.Slag_Cu_Percentage = req_data['presetParameter']['slagCuPercentage']
    args.Slag_Fe_Percentage = req_data['presetParameter']['slagFePercentage']
    args.Slag_S_Percentage = req_data['presetParameter']['slagSPercentage']
    args.Slag_SiO2_Percentage = req_data['presetParameter']['slagSiO2Percentage']

    #For GA-par
    args.epoch = req_data['modelWeight']['gaEpoch']
    args.pop = int(int(req_data['modelWeight']['gaPop']/2)*2)
    args.alpha = req_data['modelWeight']['modelFactorAlpha']
    args.beta = req_data['modelWeight']['modelFactorBeta']
    args.gama = req_data['modelWeight']['modelFactorGamma']

    #For ELEMENTS
    elements = {} 
    priorities = []
    for i in req_data['elementsTargetList']: 
        elements.update({i['name']:[i['percentage']]})
        try:
            priorities.append(i['priority'])
        except:
            priorities.append(0)
    if len(priorities) != 0:
        args.ELEMENT_MATTERS = [i['name'] for i in req_data['elementsTargetList']]
        args.ELEMENT_TARGETS_MEAN = pd.DataFrame(elements)   #a pandas from dict
        args.ELEMENT_MATTERS = list(pd.DataFrame(elements).columns)
        args.ELEMENT_PRIORITIES_SCORE = np.clip(sum(priorities) - np.array(priorities), 1, 99)
        args.ELEMENT_PRIORITIES_SCORE = args.ELEMENT_PRIORITIES_SCORE/sum(args.ELEMENT_PRIORITIES_SCORE)
    else:
        args.gama = 0  #没给优先级，则权重无效，随意填充一些Target
        args.ELEMENT_TARGETS_MEAN = pd.read_csv("../data/2_ELEMENT_TARGETS.csv")
        args.ELEMENT_MATTERS = args.ELEMENT_MATTERS
        args.ELEMENT_PRIORITIES_SCORE = np.array([0]*len(args.ELEMENT_MATTERS))
    args = compelete_basic_args(args)

    #Call GA:
    if args.threads == 1:   #for single thread debug
        constraint_eq, constraint_ueq = get_constraints(args)
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.01]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.003, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=sum(args.INGREDIENT_MUST_WITH_RATIO['ratio']), columns_just_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
        ga.run()
        sys.exit()
    _best_ratio_adjust_, _y_, best_solution, element_output = run_opt(args)
    raw_ratio = best_solution['ratio']
    best_solution = best_solution.loc[best_solution.ratio!=0]
    #if 'volume_of_storage' not in best_solution.columns:
    #    best_solution = pd.concat((best_solution, args.INGREDIENT_STORAGE.loc[best_solution.index, args.ELEMENTS+['volume_of_storage']]), axis=1)
    #best_solution = pd.concat((best_solution, args.INGREDIENT_STORAGE.loc[best_solution.index, ['number']]), axis=1)
    best_solution = web_ratio_int(best_solution)
    best_solution, element_output = mixing(args, best_solution)
    best_solution = web_consumption_int(best_solution)
    _, _y_ = evaluation(args, best_solution, element_output)

    #计算氧料比：
    args.Quality = req_data['presetParameter']['matteTargetGradePercentage']/100
    args.Slag_Cu = req_data['presetParameter']['slagCuPercentage']/100
    args.Slag_S = req_data['presetParameter']['slagSPercentage']/100
    args.Slag_Fe = req_data['presetParameter']['slagFePercentage']/100
    args.Slag_SiO2 = req_data['presetParameter']['slagSiO2Percentage']/100
    oxygenMaterialRatio, Matte_T, Slag_T, Wind_Flux, SiO2_T = calc_oxygen(args, element_output)
    #pandas to req_data
    res_element = pd_to_res(element_output)[0]
    res_data = pd_to_res(best_solution)
    new_res_element = [] 
    for key in res_element.keys(): 
        if key=='name':continue 
        new_res_element.append({'name':key, 'percentage':np.round(res_element[key], 2)})
    S_vs_Cu = res_element['S']/res_element['Cu']

    res = {
        "list": 
            res_data,
        "calculateParameter":
        {
            "oxygenMaterialRatio": round(oxygenMaterialRatio, 2),
            "totalConsumedAmount": round(sum(best_solution['consumed_amounts']), 2),
            "totalLeftOver": round(sum(best_solution['leftover']), 2),
            "best_y": round(_y_, 2),
            "paFlow": round(Wind_Flux, 2),
            "SCuRatio": round(S_vs_Cu, 2),
            "totalMatte": round(Matte_T, 2),
            "totalSlag": round(Slag_T, 2),
            "totalQuartz": round(SiO2_T, 2),
        },
        "elementsMixtureList": 
            new_res_element
    }
    return jsonify(res)

@app.route('/api/getInventory', methods=['GET'])
@cross_origin()
def getInventory():    #API 0-1   NOTE: temporary 演示版，实际不需要
    global args
    #获取库存 for 显示
    inventory_storage = get_storage(for_show=True)
    #pandas to res_data
    res_data = pd_to_res(inventory_storage)
    res_data = {
        "list": 
            res_data,
        'materialList': 
            compute_element_overview(args, inventory_storage)
    }
    return jsonify(res_data)


if __name__ == '__main__':
    doc = 'GA搜索和“三种必选（仅必选，必选且用完，必选且比例）”的关系：只有“仅必选”参与GA搜索，同时GA的5%阈值考虑“必选且比例”，原则上“必选且用完”和“必选且比例”在GA外生成solution时候才被加入，evaluation的时候“必选且用完”要另外单独考虑。'
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--DEBUG', action='store_true', default=False)
    parser.add_argument("-S", '--ON_SERVER', action='store_false', default=True)
    parser.add_argument("-O", '--OBJ', type=int, default=1)
    parser.add_argument("-E", '--epoch', type=int, default=50)
    parser.add_argument("-P", '--pop', type=int, default=100)
    parser.add_argument("-A", '--alpha', type=int, default=1)
    parser.add_argument("-B", '--beta', type=int, default=1)
    parser.add_argument("-G", '--gama', type=int, default=1)  #default=3~4  ~=2*alpha+1*beta
    parser.add_argument("-T", '--threads', type=int, default=min(2, int(cpu_count()/2)))
    parser.add_argument("-M", '--MAX_TYPE_TO_SEARCH', type=int, default=4)
    parser.add_argument("--NOT_COMPUTE", type=list, default=['渣精矿烟灰'])
    parser.add_argument('--Flow', type=int, default=150)
    parser.add_argument('--Fe2O3_vs_FeO', type=float, default=0.4)
    parser.add_argument("-ELEMENTS", '--ELEMENTS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O', 'Sb', 'Bi', 'Ni', 'Ag', 'Au'])
    parser.add_argument("-ELEMENT_MATTERS", '--ELEMENT_MATTERS', type=list, default=['Cu', 'As'])
    parser.add_argument('--OXYGEN_CONCENTRATION', type=float, default=0.85)
    parser.add_argument('--COAL_T', type=float, default=1.5)
    parser.add_argument('--OXYGEN_PEER_COAL', type=float, default=1100)
    parser.add_argument('--Fe_vs_SiO2', type=float, default=2)
    args = parser.parse_args()
    args.ELEMENT_PRIORITIES_SCORE = np.array([1]*len(args.ELEMENT_MATTERS))
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manager = Manager()
    normed_dict = manager.dict()
    normed_dict['normed_obj_amount'] = manager.list()
    normed_dict['normed_obj_leftover'] = manager.list()
    normed_dict['normed_obj_leftover_01'] = manager.list()
    normed_dict['normed_obj_elements'] = manager.list()
    normed_dict['normed_obj_elements_01'] = manager.list()


    if args.ON_SERVER:
        app.run(host='0.0.0.0', port=7001, debug=True)
    else:
        #Mannual:
        #获取元素配比目标
        args.ELEMENT_TARGETS_MEAN = pd.read_csv("../data/2_ELEMENT_TARGETS.csv")
        args = compelete_basic_args(args)

        #1.获取目前的solution
        SOLUTION = load_solution()
        #2.根据目前的SOLUTION混合，得到混合结果
        this_solution = SOLUTION
        #this_solution, element_output = mixing(args, this_solution)
        #3.评判标准
        #_, scores = evaluation(args, this_solution, element_output)
        #print('\n', element_output, "\n>>>THEORITICAL BEST SOLUTION YIELDS<<<:", np.round(scores,4))
        #sys.exit()

        #Random search:
        random_search = True
        #random_search = False
        if random_search:
            run_rand(args)
        sys.exit()

        #Optimization:
        #run_opt(args)



