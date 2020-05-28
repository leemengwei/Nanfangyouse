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
import torch
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

def schaffer(p):
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    print(x1, x2,   x)
    return x

def get_constraints(args):   #Constraints are weak.
    #For eq
    string_eq = "100" 
    for i in range(args.NUM_OF_TYPES_FOR_GA):
        string_eq += " - x[%s]"%i 
    constraint_eq = [
        lambda x: eval(string_eq)  # lambda x: 100 - x[0] - x[1] - x[2] - x[3],   即=0   
        ]

    #For ueq 1
    string_ueq1 = ''
    for i in range(args.NUM_OF_TYPES_FOR_GA): 
        string_ueq1 += "lambda x: 5 - x[%s],"%i     #lambda x: 5 - x[0], etc....，即x0, x1, x2 ... >5
    string_ueq1 = string_ueq1.strip(',')
    #For ueq 2
    string_ueq2 = ''
    for i in range(args.NUM_OF_TYPES_FOR_GA):
        string_ueq2 += "x[%s],"%i
    string_ueq2 = string_ueq2.strip(',')
    string_ueq2 = "lambda x: sum(np.array([%s])) - %s"%(string_ueq2, args.NUM_OF_TYPES_FOR_GA)     #即sum(np.array([x[0], x[1], x[2]....])>0)<=4
    constraint_ueq = list(eval(string_ueq1)) #  + [eval(string_ueq2)]    #两个不等式限制,目前不能加第二个

    constraint_eq = []   #加和小于100, 不注释则清空限制
    constraint_ueq = []   #不注释则清空ueq constraint, 使用上下限lb ub 5-100就可以不注释
    return constraint_eq, constraint_ueq

def adjust_ratio(args, their_ratio):
    their_ratio = their_ratio*(1-sum(args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST].ratio)) #GA生成的概率sum是100%，在mix之前调整一下
    return their_ratio

def GAwrapper(their_ratio):   #their_ratio是遗传算法给过来的, GA算法本身的API要求
    global args
    their_ratio = adjust_ratio(args, their_ratio)
    global this_solution
    this_solution = generate_solution(their_ratio)
    this_solution, element_output = mixing(args, this_solution)
    _, scores = evaluation(args, this_solution, element_output)
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
    if not (ingredient_storage[ingredient_storage.required==0]['ratio'].values==0).all():
        print("WARNING: Not must but giving a ratio??")
    if not (ingredient_storage[ingredient_storage.required==1]['ratio'].values!=0).all():
        print("WARNING: Lack of ratio for must!")
    if not sum(ingredient_storage[ingredient_storage.required==1]['ratio'].values)<1:
        print("WARNING: Error giving ratio!")
    return ingredient_storage

def get_ELEMENT_TARGETS(args):
    args.ELEMENT_TARGETS_LOW = args.ELEMENT_TARGETS_MEAN - args.ELEMENT_TARGETS_MEAN*0.1
    args.ELEMENT_TARGETS_HIGH = args.ELEMENT_TARGETS_MEAN + args.ELEMENT_TARGETS_MEAN*0.1
    return args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_MEAN, args.ELEMENT_TARGETS_HIGH

def load_solution():
    SOLUTION = pd.read_csv("../data/3_SOLUTION.csv", index_col='name')
    which_is_percentage = np.where(SOLUTION.columns=='ratio')[0][0]
    #str to percentage
    for row_idx,row in enumerate(SOLUTION.iterrows()):
        SOLUTION.iloc[row_idx, which_is_percentage] = float(row[1]['ratio'].strip("%"))/100
    return SOLUTION

def generate_solution(their_ratio):
    this_solution = args.INGREDIENT_STORAGE.reindex(index=args.INGREDIENT_CHOOSE_FROM, columns=['ratio'])
    this_solution['ratio'] = their_ratio
    #加上必备行
    this_solution = add_must(this_solution)
    return this_solution

def add_must(this_solution):
    #加上必备的行
    for must_this in args.INGREDIENT_MUST:
        this_solution.loc[must_this, 'ratio'] = args.INGREDIENT_STORAGE.loc[must_this, 'ratio']
    return this_solution

def mixing(args,this_solution):
    if np.round(this_solution['ratio'].sum(), 3) != 1:
        if args.DEBUG:print("***Warning for ratio...", this_solution['ratio'].sum())
        this_solution['ratio'] = this_solution['ratio']/this_solution['ratio'].sum()
    element_output = pd.DataFrame(np.array([0]*len(args.ELEMENTS)).reshape(1,-1), columns = args.ELEMENTS)
    for this_type in this_solution.index:
        element_output += this_solution.loc[this_type, 'ratio'] * args.INGREDIENT_STORAGE.loc[this_type][args.ELEMENTS]
    first_insufficient_type = this_solution.index[np.where(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']==min(args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']/this_solution['ratio']))[0][0]]
    consumed_amount = (this_solution['ratio']/this_solution.loc[first_insufficient_type, 'ratio'])*args.INGREDIENT_STORAGE.loc[first_insufficient_type, 'volume_of_storage']
    #after consumed, leftovers are:
    this_solution['consumed_amount'] = consumed_amount
    this_solution['leftover'] = args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'] - consumed_amount
    this_solution['consumed_amount'] = np.round(this_solution['consumed_amount'].astype(float), 2)
    this_solution['leftover'] = np.round(this_solution['leftover'].astype(float), 2)
    return this_solution, element_output

def evaluation(args, this_solution, element_output):
    #根据混合结果得到Objectives:
    obj_consumed = this_solution['consumed_amount']             #越大越好
    obj_leftover = this_solution['leftover']     #越小越好， 平滑
    obj_leftover_01 =  (this_solution['leftover']/args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage']<0.01).sum()     #越大越好, 非平滑, 少于百分之一就算0
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
    tmp = list(args.INGREDIENT_STORAGE['volume_of_storage'])
    tmp.sort()
    volume_normer = sum(tmp[-args.MAX_TYPE_ALLOWED:])

    #Objectives无量纲化：
    normed_obj_amount = obj_consumed.sum()/volume_normer    #用库存最多的4种总量做标准化  0~1, -->1
    normed_obj_leftover = 1 - obj_leftover.sum()/args.INGREDIENT_STORAGE.loc[this_solution.index, 'volume_of_storage'].sum()   #用这4类数量做标准化  0~1, -->1
    normed_obj_leftover_01 = obj_leftover_01/args.MAX_TYPE_ALLOWED   #用种类个数做标准化  0~1, -->1
    normed_obj_elements = 1 - 0.01*(obj_element_diff*args.ELEMENT_PRIORITIES_SCORE).values.sum()/max(1, len(args.ELEMENT_MATTERS))    #用需要检查的元素数量做标准化 0~1, -->1
    normed_obj_elements_01 = obj_element_01/len(args.ELEMENT_MATTERS)   #用MATTERS的ELEMENT个数做标准化 0~1, -->1

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

def run_opt_map(struct):     #多线程调用GA的map
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
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.0001]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_ALLOWED=args.MAX_TYPE_ALLOWED, ratio_taken=args.INGREDIENT_STORAGE.loc[args.INGREDIENT_MUST,'ratio'].sum())
    best_gax, best_gay = ga.run()
    best_ratio = best_gax/best_gax.sum()
    best_solution = copy.deepcopy(this_solution)
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM, 'ratio'] = best_ratio
    print("Best solution found:\n", best_solution)
    print('Best_ratio:', best_ratio, '\n', 'best_y:', best_gay)

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    return best_gay, best_ratio, best_solution

def run_rand(args):
    print("\n\nRandom search GA, 1 iters, all pop.")
    best_ys = []
    for i in range(args.threads):
        constraint_eq, constraint_ueq = get_constraints(args)
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop*args.epoch, max_iter=1, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.0001]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.01, MAX_TYPE_ALLOWED=args.MAX_TYPE_ALLOWED)
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
    rs = pool.map(run_opt_map, struct_list)              #CORE
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
    best_adjust_ratio = adjust_ratio(args, best_ratios[best_ys.argmin()])   #记得调整一下
    best_solution = best_solutions[best_ys.argmin()]
    best_solution.loc[args.INGREDIENT_CHOOSE_FROM, 'ratio'] = best_adjust_ratio
    best_y = best_ys[best_ys.argmin()]
    print("***BEST:", best_solution)
    print(best_ys.min())
    _, element_output = mixing(args, best_solution)
    return best_adjust_ratio, best_solution, element_output

#For server~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pd_to_res(storage):
    res_data = [] 
    key_copying = {'ratio':'calculatePercentage', 'leftover':'inventoryBalance', 'volume_of_storage':'inventory'}
    for i in storage.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
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
    args.INGREDIENT_CHOOSE_FROM = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required==0].index)
    args.INGREDIENT_MUST = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index)
    args.NUM_OF_TYPES_FOR_GA = len(args.INGREDIENT_CHOOSE_FROM)

    args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_MEAN, args.ELEMENT_TARGETS_HIGH = get_ELEMENT_TARGETS(args)
    return args

@app.route('/api/calculate', methods=['POST', 'GET'])
@cross_origin()
def calculate():
    req_data = request.get_json()
    print(req_data, file=stderr)
    global args

    #req_data to pandas:
    args.INGREDIENT_STORAGE = pd.DataFrame() 
    for i in req_data['list']: 
        args.INGREDIENT_STORAGE = args.INGREDIENT_STORAGE.append(pd.DataFrame(data=i, index=[i['name']]))
    args.alpha = req_data['presetParameter']['modelFactorAlpha']
    args.beta = req_data['presetParameter']['modelFactorBeta']
    args.gama = req_data['presetParameter']['modelFactorGamma']
    args.MAX_TYPE_ALLOWED = req_data['presetParameter']['maxType']
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
        args.ELEMENT_PRIORITIES_SCORE = np.clip(len(args.ELEMENT_MATTERS) - np.array(priorities), 1, 99)
    else:
        args.gama = 0  #没给优先级，则权重无效，随意填充一些Target
        args.ELEMENT_TARGETS_MEAN = pd.read_csv("../data/2_ELEMENT_TARGETS.csv")
        args.ELEMENT_MATTERS = args.ELEMENT_MATTERS
        args.ELEMENT_PRIORITIES_SCORE = np.array([0]*len(args.ELEMENT_MATTERS))
    args = compelete_basic_args(args)
    args.epoch = req_data['presetParameter']['gaEpoch']
    args.pop = int(int(req_data['presetParameter']['gaPop']/2)*2)

    #Call GA:
    _best_ratio_adjust_, best_solution, element_output = run_opt(args)
    best_solution.ratio = np.round(best_solution.ratio, 2)
    best_solution = best_solution.loc[best_solution.ratio!=0]
    best_solution = pd.concat((best_solution, args.INGREDIENT_STORAGE.loc[best_solution.index, args.ELEMENTS+['volume_of_storage','number']]), axis=1)
 
    #计算氧料比：
    Quality = req_data['presetParameter']['matteTargetGradePercentage']
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
    parser.add_argument("-M", '--MAX_TYPE_ALLOWED', type=int, default=4)
    parser.add_argument("-ELEMENTS", '--ELEMENTS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    parser.add_argument("-ELEMENT_MATTERS", '--ELEMENT_MATTERS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    args = parser.parse_args()
    args.ELEMENT_PRIORITIES_SCORE = np.array([1]*len(args.ELEMENT_MATTERS))
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Start....")

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
        this_solution, element_output = mixing(args, this_solution)
        #3.评判标准
        _, scores = evaluation(args, this_solution, element_output)
        print('\n', element_output, "\n>>>THEORITICAL BEST SOLUTION YIELDS<<<:", np.round(scores,4))
        #sys.exit()

        #Random search:
        #random_search = True
        random_search = False
        if random_search:
            run_rand(args)

        #Optimization:
        #run_opt(args)



