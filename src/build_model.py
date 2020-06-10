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
    their_ratio = their_ratio*(1-sum(args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_WITH_RATIO].ratio)) #GA生成的概率sum是100%，但有时可能有“必选且指定比例”项目存在，GA内部仅在5%阈值上是考虑了这个因素的，所以在mix之前调整一下
    return their_ratio

def GAwrapper(their_ratio):   #their_ratio是遗传算法给过来的, GA算法本身的API要求, TODO:居然their_ratio是一个个给回来的，准备矢量化吧！
    global args
    their_ratio = adjust_GA_ratio(args, their_ratio)
    global this_solution
    this_solution = generate_solution(their_ratio)
    this_solution, element_output = mixing(args, this_solution)
    global obj_dict
    obj_dict, scores = evaluation(args, this_solution, element_output)
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
    which_is_time = np.where(ingredient_storage.columns=='when_comes_in')[0][0]
    #str to datetime
    for row_idx,row in enumerate(ingredient_storage.iterrows()):
        ingredient_storage.iloc[row_idx, which_is_time] = datetime.datetime.strptime(row[1].when_comes_in, "%Y/%m/%d %H:%M")
    return ingredient_storage

def get_ELEMENT_TARGETS(args):
    args.ELEMENT_TARGETS_LOW = args.ELEMENT_TARGETS_MEAN - args.ELEMENT_TARGETS_MEAN*0.01
    args.ELEMENT_TARGETS_HIGH = args.ELEMENT_TARGETS_MEAN + args.ELEMENT_TARGETS_MEAN*0.01
    return args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_MEAN, args.ELEMENT_TARGETS_HIGH

def load_solution():
    SOLUTION = pd.read_csv("../data/3_SOLUTION.csv", index_col='name')
    which_is_percentage = np.where(SOLUTION.columns=='ratio')[0][0]
    #str to percentage
    for row_idx,row in enumerate(SOLUTION.iterrows()):
        SOLUTION.iloc[row_idx, which_is_percentage] = float(row[1]['ratio'].strip("%"))/100
    return SOLUTION

def generate_solution(their_ratio):
    part_solution = args.INGREDIENT_STORAGE.reindex(index=args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN, columns=['ratio'])
    part_solution['ratio'] = their_ratio
    #加上必备行
    part_solution = add_solution_with_must_ratio_only(part_solution)  #GA出来已有‘仅必选’&‘必清空’
    return part_solution

def add_solution_with_must_ratio_and_must_clean(part_solution):    #TODO决定不走这条路了，还是把‘仅必选、必选且用光’两个都传给GA，之后对后者再惩罚吧。
    #a) GA出来已经包含了’仅必备’，这里首先考虑加上‘必备且有百分比’的行：
    for must_this in args.INGREDIENT_MUST_WITH_RATIO:
        part_solution.loc[must_this, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_this, 'ratio']
    #b) 目前仅剩下‘必备清仓’未考虑了，加上"必备清仓"，这里稍复杂，需要在本处根据GA输出各项目的比例，动态给出清仓项比例。
    tmp_consumption = get_consumed_amounts(part_solution)
    consumption_with_must_clean = sum(tmp_consumption) + sum(args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_CLEAN,'volume_of_storage'])
    for must_clean in args.INGREDIENT_MUST_CLEAN:
        part_solution.loc[must_clean, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_clean, 'volume_of_storage']/consumption_with_must_clean
    #c) 再根据消耗量反算其它项目，但显而易见这会影响到其他原为5%的项目，导致他们的比例更低：  #TODO：感觉不好解决
    part_solution.loc[tmp_consumption.index, 'ratio'] = tmp_consumption/consumption_with_must_clean
    full_solution = part_solution
    return full_solution

def add_solution_with_must_ratio_only(part_solution):  #走这条路的话，注意需要确认已经把‘仅必选、必选且用光’两个都传给GA了，则此处只补充‘必备且有百分比’的项目。
    for must_this in args.INGREDIENT_MUST_WITH_RATIO:
        part_solution.loc[must_this, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_this, 'ratio']  #TODO
    full_solution = part_solution
    return full_solution

def get_consumed_amounts(this_solution):
    try:
        first_insufficient_type = this_solution.index[np.where(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']==min(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']))[0][0]]
    except:
        print("*"*888)
        print(args.INGREDIENT_STORAGE.loc[this_solution.index,'volume_of_storage']/this_solution['ratio']==min(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']))
        embed()
    consumed_amounts = (this_solution['ratio']/this_solution.loc[first_insufficient_type, 'ratio'])*args.INGREDIENT_STORAGE.loc[first_insufficient_type, 'volume_of_storage']
    return consumed_amounts

def mixing(args, this_solution):
    if args.MAX_TYPE_TO_SEARCH != 0:
        if not this_solution.loc[args.INGREDIENT_CHOOSE_FROM[0], 'ratio'] < np.inf:    #当待选1个，precision=1,GA内部会出现唯一一个取为0情况导致出现nan，修订为1-ratio_taken.
            print("*"*88)
            this_solution.loc[args.INGREDIENT_CHOOSE_FROM[0], 'ratio'] = 1 - args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_WITH_RATIO,'ratio'].sum()
    if np.round(this_solution['ratio'].sum(), 3) != 1:
        if args.DEBUG:print("***Warning for ratio...", this_solution['ratio'].sum())
        this_solution['ratio'] = this_solution['ratio']/this_solution['ratio'].sum()
    consumed_amounts = get_consumed_amounts(this_solution)
    element_output = pd.DataFrame(np.array([0]*len(args.ELEMENTS)).reshape(1,-1), columns = args.ELEMENTS)
    for this_type in this_solution.index:
        element_output += this_solution.loc[this_type, 'ratio'] * args.INGREDIENT_STORAGE.loc[this_type][args.ELEMENTS]
    #after consumed, leftovers are:
    this_solution['consumed_amounts'] = consumed_amounts
    this_solution['leftover'] = args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'] - consumed_amounts
    this_solution['consumed_amounts'] = np.round(this_solution['consumed_amounts'].astype(float), 2)
    this_solution['leftover'] = np.round(this_solution['leftover'].astype(float), 2)
    return this_solution, element_output

def evaluation(args, this_solution, element_output):
    evaluate_on = args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN
    #根据混合结果得到Objectives:
    obj_consumed = this_solution.loc[evaluate_on]['consumed_amounts']             #越大越好
    obj_leftover = this_solution.loc[evaluate_on, 'leftover'] #越小越好， 平滑
    obj_leftover.loc[args.INGREDIENT_MUST_CLEAN] *= 1e5    #Penalty here，必清的不清，则惩罚
    obj_leftover_01 =  (this_solution.loc[evaluate_on, 'leftover']/args.INGREDIENT_STORAGE.loc[evaluate_on, 'volume_of_storage']<0.01).sum()     #越大越好, 非平滑, 少于百分之一就算0
    obj_element_diff = abs(args.ELEMENT_TARGETS_MEAN - element_output)[args.ELEMENT_MATTERS]    #越小越好，平滑
    obj_element_01 = list(((args.ELEMENT_TARGETS_LOW[args.ELEMENT_MATTERS] < element_output[args.ELEMENT_MATTERS]) & (element_output[args.ELEMENT_MATTERS] < args.ELEMENT_TARGETS_HIGH[args.ELEMENT_MATTERS])).loc[0]).count(1)    #越大越好, 非平滑

    #记录
    obj_dict = {}
    obj_dict['obj_consumed'] = obj_consumed
    obj_dict['obj_leftover'] = obj_leftover
    obj_dict['oibj_leftover_01'] = obj_leftover_01
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
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.1]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_WITH_RATIO,'ratio'].sum(), columns_just_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
    best_gax, best_gay = ga.run()
    best_ratio = best_gax/best_gax.sum()
    best_solution = copy.deepcopy(this_solution)
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN, 'ratio'] = best_ratio

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
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop*args.epoch, max_iter=1, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.1]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_WITH_RATIO,'ratio'].sum(), columns_just_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
        best_gax, best_gay = ga.run()
        best_ys.append(best_gay[0])
    best_ys = np.array(best_ys)
    print("***Random search best mean:", best_ys.mean(), best_ys.min())
    sys.exit()

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
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN, 'ratio'] = best_adjust_GA_ratio
    best_y = best_ys[best_ys.argmin()]
    print("***BEST:", best_solution)
    print(best_ys.min())
    _, element_output = mixing(args, best_solution)
    return best_adjust_GA_ratio, best_solution, element_output

#For server~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pd_to_res(storage):
    res_data = [] 
    key_copying = {'ratio':'calculatePercentage', 'leftover':'inventoryBalance', 'volume_of_storage':'inventory'}
    for i in storage.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
            if key_attr == 'required':   #这个key特殊处理一下true false
                this_dict[key_attr] = True if i[1][this_attr] == 1 else False
            else:
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
        args.INGREDIENT_STORAGE = args.INGREDIENT_STORAGE
    args.INGREDIENT_MUST_WITH_RATIO = list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio!=0].index))   #必选比例以定
    args.INGREDIENT_MUST_CLEAN = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean!=0].index)  #必选且必须清空该料
    args.INGREDIENT_JUST_MUST = list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio==0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean==0].index))  #必选但不指定比例
    args.INGREDIENT_CHOOSE_FROM = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required==0].index)
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST = args.INGREDIENT_CHOOSE_FROM + args.INGREDIENT_JUST_MUST
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_WITH_RATIO = args.INGREDIENT_CHOOSE_FROM + args.INGREDIENT_JUST_MUST + args.INGREDIENT_MUST_WITH_RATIO
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_CHOOSE_FROM + args.INGREDIENT_JUST_MUST + args.INGREDIENT_MUST_CLEAN
    args.JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_JUST_MUST + args.INGREDIENT_MUST_CLEAN
    #整理一下顺序
    order = list(args.INGREDIENT_STORAGE.index).index
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST.sort(key=order)
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_WITH_RATIO.sort(key=order)
    args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.sort(key=order)
    args.NUM_OF_TYPES_FOR_GA = len(args.INGREDIENT_CHOOSE_FROM) + len(args.INGREDIENT_JUST_MUST) + len(args.INGREDIENT_MUST_CLEAN)
    args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_MEAN, args.ELEMENT_TARGETS_HIGH = get_ELEMENT_TARGETS(args)
    if len(args.INGREDIENT_MUST_CLEAN)>0:
        dimension_reducer = args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_CLEAN, 'volume_of_storage']/min(args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST_CLEAN, 'volume_of_storage'])
    else:
        dimension_reducer = pd.DataFrame([])
    args.DIMENSION_REDUCER_DICT = {} 
    for i in dimension_reducer.index: 
        args.DIMENSION_REDUCER_DICT[args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index(i)] = dimension_reducer[i]
    #另外需要给ga准备just must(和must clean)的col index
    args.JUST_MUST_AND_MUST_CLEAN_COLUMNS = []
    for i in args.JUST_MUST_AND_MUST_CLEAN:
        args.JUST_MUST_AND_MUST_CLEAN_COLUMNS.append(args.INGREDIENT_CHOOSE_FROM_AND_JUST_MUST_AND_MUST_CLEAN.index(i))
    return args

@app.route('/api/calculate', methods=['POST', 'GET'])
@cross_origin()
def calculate():
    req_data = request.get_json()
    global args

    #req_data to pd:
    args.INGREDIENT_STORAGE = pd.DataFrame()
    for i in req_data['list']: 
        args.INGREDIENT_STORAGE = args.INGREDIENT_STORAGE.append(pd.DataFrame(data=i, index=[i['name']]))
    args.INGREDIENT_STORAGE.required = args.INGREDIENT_STORAGE.required+0  #这样就把网页传回来的true false改成 01 了
    args.INGREDIENT_STORAGE['ratio'] = args.INGREDIENT_STORAGE['calculatePercentage']  #这样就把网页传回来的calcPrecent 改成了ratio
    args.epoch = req_data['presetParameter']['gaEpoch']
    args.pop = int(int(req_data['presetParameter']['gaPop']/2)*2)
    args.alpha = req_data['presetParameter']['modelFactorAlpha']
    args.beta = req_data['presetParameter']['modelFactorBeta']
    args.gama = req_data['presetParameter']['modelFactorGamma']
    args.MAX_TYPE_TO_SEARCH = req_data['presetParameter']['maxType']
    elements = {} 
    priorities = []
    for i in req_data['elementsTargetList']: 
        elements.update({i['name']:[i['percentage']]})
        try:
            priorities.append(i['priority'])
        except:
            priorities.append(0)
    if len(priorities) != 0:
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
    _best_ratio_adjust_, best_solution, element_output = run_opt(args)
    best_solution = best_solution.loc[best_solution.ratio!=0]
    best_solution = pd.concat((best_solution, args.INGREDIENT_STORAGE.loc[best_solution.index, args.ELEMENTS+['volume_of_storage','number']]), axis=1)
    #Web Show INT:
    for index,content in best_solution.iterrows():   #如果页面上想展示detial：
        best_solution.loc[index, 'ratio'] = str(np.round(best_solution.loc[index, 'ratio'],2))+" (%s%%)"%np.round(best_solution.loc[index, 'ratio']*100,2)
    #best_solution.ratio = np.round(best_solution.ratio, 2)
    best_solution.consumed_amounts = np.round(best_solution.consumed_amounts)
    best_solution.leftover = np.round(best_solution.leftover)

    #计算氧料比：
    Quality = req_data['presetParameter']['matteTargetGradePercentage']/100
    Slag_Cu = req_data['presetParameter']['slagCuPercentage']/100
    Slag_S = req_data['presetParameter']['slagSPercentage']/100
    Slag_Fe = req_data['presetParameter']['slagFePercentage']/100
    Slag_SiO2 = req_data['presetParameter']['slagSiO2Percentage']/100
    Flow = 150
    Fe2O3_vs_FeO = 0.4
    oxygenMaterialRatio = calc_oxygen(element_output, Quality, Slag_Cu, Slag_S, Slag_Fe, Slag_SiO2, Flow, Fe2O3_vs_FeO)

    #pandas to req_data
    res_element = pd_to_res(element_output)[0]
    res_data = pd_to_res(best_solution)
    new_res_element = [] 
    for key in res_element.keys(): 
        if key=='name':continue 
        new_res_element.append({'name':key, 'percentage':np.round(res_element[key], 2)})

    res_data = {
        "list": 
            res_data,
        "calculateParameter":
        {
            "oxygenMaterialRatio": str(oxygenMaterialRatio),
        },
        "elementsMixtureList": 
        new_res_element
    }
    return jsonify(res_data)

@app.route('/api/getInventory', methods=['GET'])
@cross_origin()
def getInventory():
    global args
    #获取库存 for 显示
    inventory_storage = get_storage(for_show=True)
    #pandas to res_data
    res_data = pd_to_res(inventory_storage)

    return jsonify(res_data)


if __name__ == '__main__':
    doc = 'GA搜索和“三种必选（仅必选，必选且用完，必选且比例）”的关系：只有“仅必选”参与GA搜索，同时GA的5%阈值考虑“必选且比例”，原则上“必选且用完”和“必选且比例”在GA外生成solution时候才被加入，evaluation的时候“必选且用完”要另外单独考虑。'
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--DEBUG', action='store_true', default=False)
    parser.add_argument("-S", '--ON_SERVER', action='store_true', default=False)
    parser.add_argument("-O", '--OBJ', type=int, default=1)
    parser.add_argument("-E", '--epoch', type=int, default=50)
    parser.add_argument("-P", '--pop', type=int, default=100)
    parser.add_argument("-A", '--alpha', type=int, default=1)
    parser.add_argument("-B", '--beta', type=int, default=1)
    parser.add_argument("-G", '--gama', type=int, default=1)
    parser.add_argument("-T", '--threads', type=int, default=int(cpu_count()/2))
    parser.add_argument("-M", '--MAX_TYPE_TO_SEARCH', type=int, default=4)
    parser.add_argument("-ELEMENTS", '--ELEMENTS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    #parser.add_argument("-ELEMENT_MATTERS", '--ELEMENT_MATTERS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    parser.add_argument("-ELEMENT_MATTERS", '--ELEMENT_MATTERS', type=list, default=['Cu', 'As'])
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

        #Optimization:
        run_opt(args)



