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
from flask_cors import cross_origin  #WEB
from calc_oxygen import calc_oxygen
app = Flask(__name__)
import random
epsilon = 1e-10
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


def shrink_GA_ratio(their_ratio):  #Vectorized
    their_ratio = their_ratio*(1-args.SHRINGKER) #GA生成的概率sum是100%，但有时可能有“必选且指定比例”项目存在，GA内部仅在5%阈值上是考虑了这个因素的，生成的ratio sum会是100%，这里收缩为95%，随后再加5%，mix之前调整一下
    return their_ratio

def GAwrapper(their_ratio):   #their_ratio是遗传算法给过来的, GA算法本身的要求, NOTE:their_ratio是一个个给回来的，准备矢量化
    #start = time.time()
    #global args
    #GA搜索的时候生成ratio和为100%，为了获得实际的evaluate，需要每次先缩再加must with ratio项（和最终的gax best 无关，gax best到时候还要再加）
    their_ratio = shrink_GA_ratio(their_ratio)
    #global full_solution
    full_solution = expand_full_solution(their_ratio)   
    #time_1 = time.time()
    full_solution, element_output = mixing(args, full_solution)
    #time_2 = time.time()
    global obj_dict
    obj_dict, scores = evaluation(args, full_solution, element_output)
    #time_3 = time.time()
    #print('1:',100*(time_1-start)/(time.time()-start), '%')
    #print('2:',100*(time_2-time_1)/(time.time()-start), '%')
    #print('3:',100*(time_3-time_2)/(time.time()-start), '%')
    #print('ALl:', time.time()-start)
    return scores

def C(n,k):  
    #import operator
    #return reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))  
    out = comb(n, k)
    return out

def get_storage(for_web=False):
    if for_web:
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
    which_is_percentage = np.where(SOLUTION.columns=='calculatePercentage')[0][0]
    #str to percentage
    for row_idx,row in enumerate(SOLUTION.iterrows()):
        SOLUTION.iloc[row_idx, which_is_percentage] = float(row[1]['calculatePercentage'].strip("%"))/100
    return SOLUTION

def expand_full_solution(their_ratio):
    #加上必备行
    args.INGREDIENT_FOR_GA.calculatePercentage = their_ratio
    full_solution = args.INGREDIENT_FOR_GA.append(args.INGREDIENT_MUST_WITH_RATIO)
    full_solution = full_solution.reindex(args.INGREDIENT_STORAGE.index)
    return full_solution  

def get_consumed_amounts(_ratios_, _volume_of_storage_, _solution_index_):  #Vectorized
    _pos_ = np.argmin((_volume_of_storage_+epsilon)/(_ratios_+epsilon**2))
    #first_insufficient_type = _solution_index_[np.where(_volume_of_storage_/_ratios_==min(_volume_of_storage_/_ratios_))[0][0]]
    consumed_amounts = (_ratios_/_ratios_[_pos_])*_volume_of_storage_[_pos_]
    return consumed_amounts

def mixing(args, full_solution):
    _ratios_ = full_solution['calculatePercentage'].values
    _volume_of_storage_ = full_solution['inventory'].values
    _solution_index_ = full_solution.index
    _elements_ = full_solution[args.ELEMENTS].values
    if not _ratios_[0] < np.inf:    #当待选1个，precision=1,GA内部会出现唯一一个取为0情况导致出现nan，修订为1-ratio_taken.
        _ratios_[0] = 1 - sum(args.INGREDIENT_MUST_WITH_RATIO['calculatePercentage'].values)
    if np.round(_ratios_.sum(), 3) != 1:
        if args.DEBUG:print("***Warning for ratio...", _ratios_.sum())
        _ratios_ = _ratios_/_ratios_.sum()
    #根据实际计算消耗情况
    _consumed_amounts_ = get_consumed_amounts(_ratios_, _volume_of_storage_, _solution_index_)
    full_solution['consumed_amounts'] = _consumed_amounts_
    full_solution['inventoryBalance'] = _volume_of_storage_ - _consumed_amounts_
    full_solution['productionTime'] = np.round(_volume_of_storage_/(_ratios_*args.Flow+epsilon)/24, 1)
    #compute mix element of full solution
    _element_output_ = (_ratios_.reshape(-1,1) * _elements_).sum(axis=0)
    element_output = pd.DataFrame(_element_output_.reshape(1,-1), columns=args.ELEMENTS)
    return full_solution, element_output

def evaluation(args, full_solution, element_output):
    _volume_of_storage_ = args.INGREDIENT_STORAGE['inventory'].values
    _consumed_amounts_ = full_solution['consumed_amounts'].values
    _leftover_ = full_solution['inventoryBalance'].values
    _should_clean_storage_= full_solution['inventory'][args.INGREDIENT_MUST_CLEAN.index].values
    _should_clean_leftover_ = full_solution['inventoryBalance'][args.INGREDIENT_MUST_CLEAN.index].values
    penalty = sum(np.round(_should_clean_leftover_/_should_clean_storage_, 2)) * 1000 #以矢量的方式判断是否小于1%
    #根据混合结果得到Objectives:
    obj_consumed = _consumed_amounts_       #越大越好
    obj_leftover = _leftover_ #越小越好， 平滑
    obj_element_diff = (abs(args.ELEMENT_TARGETS_MEAN - element_output)[args.ELEMENT_MATTERS]).values    #越小越好，平滑

    #记录
    obj_dict = {}
    #obj_dict['obj_consumed'] = obj_consumed
    #obj_dict['obj_leftover'] = obj_leftover
    #obj_dict['obj_leftover_01'] = obj_leftover_01
    #obj_dict['obj_element_diff'] = obj_element_diff
    #obj_dict['obj_element_01'] = obj_element_01

    #Misc
    volume_normer = _volume_of_storage_[_volume_of_storage_.argsort()][-args.MAX_TYPE_TO_SEARCH:].sum()
    leftover_normer = _volume_of_storage_.sum()

    #Objectives无量纲化：
    normed_obj_amount = obj_consumed.sum()/volume_normer    #用库存最多N种总量做标准化(不包含必选项目)  0~1, -->1
    normed_obj_leftover = 1 - obj_leftover.sum()/leftover_normer   #用所有可选择的类数量做标准化  0~1, -->1
    normed_obj_elements = 1 - 0.01*(obj_element_diff*args.ELEMENT_PRIORITIES_SCORE).sum()    #用需要检查的元素数量做标准化 0~1, -->1
 
    #记录
    #normed_dict = {}
    #global normed_dict
    #normed_dict['normed_obj_amount'].append(normed_obj_amount)
    #normed_dict['normed_obj_leftover'].append(normed_obj_leftover)
    #normed_dict['normed_obj_leftover_01'].append(normed_obj_leftover_01)
    #normed_dict['normed_obj_elements'].append(normed_obj_elements)
    #normed_dict['normed_obj_elements_01'].append(normed_obj_elements_01)
    #print(normed_dict, "ele_diff:", obj_element_diff, 'priorities', args.ELEMENT_PRIORITIES_SCORE)

    #Multi-Obj to Single_obj:   #更正认识：平滑的局地极小更多，错误解可能更大, 非平滑的只有在真正正确的时候得到小值
    objective_function = 1000 - args.alpha*normed_obj_amount - args.beta*normed_obj_leftover - 3*args.gama*normed_obj_elements + penalty    #GA的适应度会是他的负值，恰好是loss最低的适应度最大。故此obj需要-->0
    score = objective_function
    if args.DEBUG:print(full_solution, '  ', score)
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
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop, max_iter=args.epoch, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.001]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.002, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=args.ratio_taken, columns_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
    best_gax, best_gay = ga.run()
    Y_history = pd.DataFrame(ga.all_history_Y)
    return best_gax, best_gay

def run_rand(args):
    print("\n\nRandom search GA, 1 iters, all pop.")
    best_ys = []
    for i in range(2): 
        constraint_eq, constraint_ueq = get_constraints(args)
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.pop*args.epoch, max_iter=1, lb=[0]*args.NUM_OF_TYPES_FOR_GA, ub=[100]*args.NUM_OF_TYPES_FOR_GA, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=[0.001]*args.NUM_OF_TYPES_FOR_GA, prob_mut=0.002, MAX_TYPE_TO_SEARCH=args.MAX_TYPE_TO_SEARCH, ratio_taken=args.ratio_taken, columns_must=[args.JUST_MUST_AND_MUST_CLEAN_COLUMNS, args.DIMENSION_REDUCER_DICT])
        best_gax, best_gay = ga.run()
        best_ys.append(best_gay[0])
    best_ys = np.array(best_ys)
    print("***Random search best mean:", best_ys.mean(), best_ys.min())

def run_opt(args):
    blobs = []
    #pool_num = int(cpu_count()/2)
    pool_num = int(cpu_count())
    print('Run times and pool num', pool_num)
    pool = Pool(processes = pool_num)   #这个固定死，效率最高,跟做多少次没关系
    struct_list = []
    for i in range(pool_num):  #同时做n次
        struct_list.append([i, args])
    rs = pool.map(run_opt_map, struct_list) #CORE
    pool.close()
    pool.join()
    ys = np.empty(0)
    ratios = np.empty((0,args.NUM_OF_TYPES_FOR_GA))
    #Re-organize results:
    for r in rs:
        ratio = r[0]
        y = r[1]
        ys = np.hstack((ys, y))
        ratios = np.vstack((ratios, ratio))
    best_one = ys.argmin()
    best_ratio = ratios[best_one] 
    best_y = ys[best_one]
    #记得先缩再放 (GA生成的概率sum是100%，但有时可能有“必选且指定比例”项目存在)
    best_shrink_ratio = shrink_GA_ratio(best_ratio) 
    best_solution = expand_full_solution(best_shrink_ratio)
    print("***BEST:", best_solution)
    print(best_y)
    _, element_output = mixing(args, best_solution)
    return best_shrink_ratio, best_y, best_solution, element_output

#For server~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pd_to_res(storage):
    res_data = [] 
    storage = storage.fillna(0)
    for i in storage.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
            #if key_attr == 'required' or key_attr == 'clean' or key_attr == 'cohesion':   #这个key特殊处理一下true false
            #    this_dict[key_attr] = True if i[1][this_attr] == 1 else False
            #else:  #貌似不特殊处理才对,特殊处理会影响可编辑性。
            this_dict[key_attr] = i[1][this_attr]
        res_data.append(this_dict) 
    return res_data 

def compelete_basic_args(args):
    args.INGREDIENT_STORAGE = args.INGREDIENT_STORAGE.fillna(0)
    #获取库存 for 计算
    if not args.ON_SERVER:
        args.INGREDIENT_STORAGE = get_storage()
    else:
        pass
    args.INGREDIENT_MUST_WITH_RATIO = args.INGREDIENT_STORAGE.loc[list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.calculatePercentage!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean==0].index))]   #必选比例以定
    args.ratio_taken = sum(args.INGREDIENT_MUST_WITH_RATIO['calculatePercentage'])
    args.INGREDIENT_MUST_CLEAN = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean!=0].index)]  #必选且必须清空该料
    args.INGREDIENT_JUST_MUST = args.INGREDIENT_STORAGE.loc[list(set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.required!=0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.calculatePercentage==0].index) & set(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.clean==0].index))]  #必选但不指定比例
    args.INGREDIENT_NO_CONDITION = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_STORAGE[(args.INGREDIENT_STORAGE.required+args.INGREDIENT_STORAGE.clean)==0].index)]
    args.INGREDIENT_FOR_GA = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_NO_CONDITION.index) + list(args.INGREDIENT_JUST_MUST.index) + list(args.INGREDIENT_MUST_CLEAN.index)]
    args.JUST_MUST_AND_MUST_CLEAN = args.INGREDIENT_STORAGE.loc[list(args.INGREDIENT_JUST_MUST.index) + list(args.INGREDIENT_MUST_CLEAN.index)]
    #整理一下顺序, 要给GA准备辅助的位置，来简化must clean和just must两个项目
    args.INGREDIENT_FOR_GA = args.INGREDIENT_FOR_GA.reindex(args.INGREDIENT_STORAGE.index).dropna()  #此处必须dropna！ 因为完整的index会给前者带来nan
    args.NUM_OF_TYPES_FOR_GA = len(args.INGREDIENT_NO_CONDITION) + len(args.INGREDIENT_JUST_MUST) + len(args.INGREDIENT_MUST_CLEAN)
    args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_HIGH = get_elements_boundary(args)
    #对于必清的项目，计算其相互的比例倍数，准备通过dimension reducer给GA算法
    if len(args.INGREDIENT_MUST_CLEAN)>0:
        dimension_reducer = args.INGREDIENT_MUST_CLEAN['inventory']/min(args.INGREDIENT_MUST_CLEAN['inventory'])
    else:
        dimension_reducer = pd.DataFrame([])
    args.DIMENSION_REDUCER_DICT = {} 
    for i in dimension_reducer.index: 
        args.DIMENSION_REDUCER_DICT[list(args.INGREDIENT_FOR_GA.index).index(i)] = dimension_reducer[i]
    #另外需要给ga准备just must(和must clean)的col index
    args.JUST_MUST_AND_MUST_CLEAN_COLUMNS = []
    for i in args.JUST_MUST_AND_MUST_CLEAN.index:
        args.JUST_MUST_AND_MUST_CLEAN_COLUMNS.append(list(args.INGREDIENT_FOR_GA.index).index(i))
    args.SHRINGKER = sum(args.INGREDIENT_MUST_WITH_RATIO['calculatePercentage'].values)
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

@app.route('/api/quick_recommend', methods=['POST', 'GET'])
@cross_origin()
def quick_recommend():   #API 3 
    try:
        req_data = request.get_json()
        req_data['presetParameter'] = req_data['presetParameter_3']
        #Web-set solution:
        solution_1 = []
        solution_2 = []
        for i in req_data['list'][0]:
            solution_1.append(i)
        for i in req_data['list'][1]:
            solution_2.append(i)
        solution_1 = req_to_pd(solution_1)
        solution_2 = req_to_pd(solution_2)
        #NOTE:传订单1（被衔接）当时的手动修改的库存状态。
        solution_1.loc[:, 'inventoryBalance'] = copy.deepcopy(solution_1['inventory'])
        solution_2.loc[:, 'inventoryBalance'] = copy.deepcopy(solution_2['inventory'])
        solution_1 = solution_1.set_index('number')
        solution_2 = solution_2.set_index('number')
        solution_1['number'] = solution_1.index
        solution_2['number'] = solution_2.index
   
        #Web-set parameters: (注意指的是新订单的各个量)
        global args  #这里要给args加冰铜参数等。
        args = get_presets(args, req_data)
    
        def get_compose_solution_from_to(solution_2, solution_1):   #Pick from 2, add into 1
            concat_solution = pd.DataFrame([])
            concat_oxygen = np.array([])
            oxygenMaterialRatio_1 = float(req_data['oxygenMaterialRatio']['formula1'])
            oxygenMaterialRatio_2 = float(req_data['oxygenMaterialRatio']['formula2'])
    
            #穷举计算所有compose solution
            oxygenMaterialRatios = []
            compose_solutions = []
            compose_consumed_amounts_sorter = []    #这个sorter仅统计未消耗光的其他项的进一步数量，越大越好
            #特殊处理用户指定的fixed part，先取出来
            solution_1_fixed_part = solution_1[solution_1['fixed'] == 1]  #诸如渣精矿可以勾选fixed
            ratio_taken = sum(solution_1['calculatePercentage'][solution_1_fixed_part.index].values)
            dropped_solution_1 = solution_1.drop(solution_1_fixed_part.index)
            solution_1_short_types = dropped_solution_1[dropped_solution_1['cohesion'] == 1].index  #让用户选择需要衔接哪一个
            solution_2_short_types = solution_2[solution_2['inventoryBalance'] <= 300].index
            solution_1_types_avaliable = list(set(dropped_solution_1.index) - set(solution_1_short_types))
            solution_2_types_avaliable = list(set(solution_2.index) - set(solution_2_short_types) - set(solution_1.index))  #NOTE: 1、2相同的项不在这个变量中考虑，下面会补充搜索‘不添加任何项’来考虑。
            if (dropped_solution_1.loc[solution_1_types_avaliable, 'inventory'] == 0).any():
                status = 'Error! 目前总消耗为0, 请检查并衔接配方1中的0库存量物料，将之标记为衔接或排除，结果无意义'
                return dropped_solution_1, oxygenMaterialRatio_1, status
            combinations_more_to_less = list(itertools.combinations(list(solution_2.loc[solution_2_types_avaliable].sort_values('inventoryBalance').index[::-1]), len(solution_1_short_types)))
            combinations_more_to_less.insert(0, '')   #添加一个空项目进来，即‘不混入任何配方2’。
            #把新的2混到旧的1中
            for i in combinations_more_to_less:
                tmp_solution_1 = dropped_solution_1.drop(solution_1_short_types)   #耗尽项空出来
                index_after_drop = list(set(tmp_solution_1.index) - set(args.NOT_COMPUTE))
                tmp_solution_1 = pd.concat([tmp_solution_1, solution_2.loc[list(i)]]) 
                print("Searching... ",i, tmp_solution_1.index)
                #随机搜索配比组合：
                for _ in tqdm.tqdm(range(int(600/(1+len(combinations_more_to_less))))):
                    tmp_solution_1.loc[:, 'calculatePercentage'] = np.random.dirichlet(range(1, len(tmp_solution_1)+1))*(1-ratio_taken)
                    #再接fixed part回去
                    tmp_full_solution_1 = pd.concat((tmp_solution_1, solution_1_fixed_part))
                    #计算每一种组合的情况
                    tmp_full_solution_1, tmp_element_output = mixing(args, tmp_full_solution_1)   #mix之后就有新的消耗列了，然后在计算混入项的理论剩余（下面)
                    tmp_oxygenMaterialRatio, tmp_Matte_T, tmp_Slag_T, tmp_Wind_Flux, tmp_SiO2_T = calc_oxygen(args, tmp_element_output)
                    #对于那些氧料比需要满足要求的，就先记下来（最终物料存量肯定会满足要求的，因为mix时用的是两个配方的剩余量）：
                    if oxygen_ok(oxygenMaterialRatio_1, oxygenMaterialRatio_2, tmp_oxygenMaterialRatio) and (tmp_full_solution_1['calculatePercentage'].sum()>0.99):   #后一个条件为处理‘全固定某个衔接，却搜索到什么都情况余料最少，则ratio不为1’
                        compose_solutions.append(copy.deepcopy(tmp_full_solution_1))
                        oxygenMaterialRatios.append(tmp_oxygenMaterialRatio)
                        compose_consumed_amounts_sorter.append(tmp_full_solution_1['consumed_amounts'][index_after_drop].sum())
            #查找结束后看是否存在可行解。随后参考旧单子，查看其所有旧物料消耗最大的情况作为解：
            if len(compose_solutions)>=1:
                concat_solution = compose_solutions[np.array(compose_consumed_amounts_sorter).argmax()]
                concat_oxygen = oxygenMaterialRatios[np.array(compose_consumed_amounts_sorter).argmax()]
                if len(concat_solution)!=len(solution_1):
                    add_ons = '（不填加任何配方2项目，仅调整非衔接项的比例即可满足继续生产需求）'
                else:
                    add_ons = ''
                status = "Okay, 搜索完毕，在所有可行解组合中找到的物料剩余较小情况解已给出，如有需求，可考虑手动微调比例" + add_ons
            else:
                status = "Error, 衔接搜索结束，暂无法满足氧料比要求，将返回原配方1，可尝试再试一次或人工衔接."
                if not (len(solution_1_short_types)<=len(solution_2_types_avaliable)):
                    status += "另外需要注意：配方1、2除去共同物料后(相同项不应存在衔接需求，会在生产中自然衔接)，2号配料单中其他较安全的可选衔接物料已经不够（生产理论剩余后<300t的不被内部算法视作可衔接项，以防影响下次配方生产）"
                print(status)
                print(solution_1, oxygenMaterialRatio_1)
                print(solution_2, oxygenMaterialRatio_2)
                return solution_1, oxygenMaterialRatio_1, status
            return concat_solution, concat_oxygen, status
        #衔接，单项混入（单新入旧），穷举所有情况及其得分
        #NOTE: 搜索效果应该达到如下目的：1、剩下的N项按照其各自剩余量确定其比例（原则上为了尽可能同时用完）；2、加或不加新项（原则上为了不影响新的订单）；3、每次衔接不考虑“衔接之后再衔接”，即当前满足了氧料比落在之间开始生产，至于二次此次衔接是否会使得下次“氧料比区间”求解困难，不再过多考虑（实际上呼应了1,我们简化认为一次衔接后剩余的都是不需要处理的小量）；
        concat_solution, concat_oxygenMaterialRatio, status = get_compose_solution_from_to(solution_2, solution_1)
        concat_solution = web_ratio_int(concat_solution)
        concat_solution, concat_element_output = mixing(args, concat_solution)
        concat_solution = web_consumption_int(concat_solution)
        concat_oxygenMaterialRatio, concat_Matte_T, concat_Slag_T, concat_Wind_Flux, concat_SiO2_T = calc_oxygen(args, concat_element_output)  
        #pandas to req_data
        res_element = pd_to_res(concat_element_output)[0]
        concat_solution.loc[:, 'manual'] = False  #just fill mannual, 不然来回切换的时候有显示bug
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
                "totalConsumedAmount": round(sum(concat_solution['consumed_amounts']), 2),
                "totalLeftOver": round(sum(concat_solution['inventoryBalance']), 2),
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
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405

@app.route('/api/quick_update2', methods=['POST', 'GET'])
@cross_origin()
def quick_update2():   #API 4
    try:
        res_data = quick_update(by_update_2=True).json
        res_data['recommended'] = '(手动调整返回)'
        return jsonify(res_data)
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405

@app.route('/api/quick_update', methods=['POST', 'GET'])
@cross_origin()
def quick_update(by_update_2=False):   #API 2 
    try:
        req_data = request.get_json()
        #每次快速更新都重新获取界面设置的生产参数
        if by_update_2:
            req_data['presetParameter'] = req_data['presetParameter_3']
        global args
        args = get_presets(args, req_data)
    
        web_solution = req_to_pd(req_data['list'])
        old_ratio = copy.deepcopy(web_solution['calculatePercentage'])
        
        #如果网页回传了adjustRatio，则接下来mix所用的ratio响应调整。
        for i in web_solution.iterrows():
            web_solution.loc[i[0], 'calculatePercentage'] = float(web_solution.loc[i[0], 'adjustRatio'])
            web_solution.loc[i[0], 'calculatePercentage'] = float(web_solution.loc[i[0], 'adjustRatio'])
        web_solution = web_ratio_int(web_solution)
        adjust_solution, element_output = mixing(args, web_solution)
        adjust_solution = web_consumption_int(adjust_solution)
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
                "totalLeftOver": round(sum(adjust_solution['inventoryBalance']), 2),
                "best_y": round(0.00,2),
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
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405


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
        pd_data.loc[:, 'calculatePercentage'] = pd_data['calculatePercentage']  #这样就把网页传回来的calcPrecent 改成了ratio
    except:
        pass
    for col in pd_data.columns:
        try:
            pd_data[col] = pd_data[col].astype(float)
        except:
            print('Not converting %s'%col)
    return pd_data

#Web Show INT:
def web_ratio_int(best_solution):
    try:
        raw_ratio = copy.deepcopy(np.round(best_solution['calculatePercentage'], 4))
        interger_ratio = np.round(best_solution['calculatePercentage'], 2)
        need_to_add = int(np.round((1-interger_ratio.sum())*100))
        if need_to_add!=0:
            #各项余数tmp
            drifts = best_solution['calculatePercentage'] - interger_ratio
            drifts_ascending = drifts.sort_values()
            if need_to_add>0: #不到100%需要补充 
                for i in range(abs(need_to_add)): 
                    interger_ratio[drifts_ascending.index[-(i+1)]] += 0.01
                    print("Adding ", drifts_ascending.index[-(i+1)])
            else:  # need_to_add<0:
                for i in range(abs(need_to_add)): 
                    interger_ratio[drifts_ascending.index[i]] -= 0.01
                    print("Cutting ", drifts_ascending.index[i])
        best_solution.loc[:, 'calculatePercentage'] = np.round(interger_ratio, 5)  #Web display bug
    except Exception as e:
        best_solution.loc[:, 'calculatePercentage'] = raw_ratio
        print("Ratio error!! pass", e)
    return best_solution

def web_consumption_int(best_solution):
    _ratios_ = best_solution['calculatePercentage'].values
    _volume_of_storage_ = best_solution['inventory'].values
    _solution_index_ = best_solution.index
    best_solution.loc[:, 'consumed_amounts'] = np.clip(np.round(get_consumed_amounts(_ratios_, _volume_of_storage_, _solution_index_), 1), 0, best_solution['inventory'])
    best_solution.loc[:, 'inventoryBalance'] = np.round(best_solution['inventory'] - best_solution['consumed_amounts'], 2)
    #for index,content in best_solution.iterrows():   #如果页面上想展示detial：
    #    best_solution.loc[index, 'calculatePercentage'] = str(best_solution.loc[index, 'calculatePercentage'])+" (%s%%)"%np.round(raw_ratio.loc[index]*100,2)
    return best_solution

def compute_element_overview(storage):
    new_res_element = []
    for this_element in args.ELEMENTS:
        new_res_element.append({'name': this_element, 'percentage': np.round(sum(storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), 'inventory']*storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), this_element]) / sum(storage.loc[list(set(storage.index) - set(args.NOT_COMPUTE)), 'inventory']), 2)})
    return new_res_element

def get_presets(args, req_data):
    args.Matte_Cu_Percentage  = float(req_data['presetParameter']['matteTargetGradePercentage'])
    args.Matte_Fe_Percentage  = float(req_data['presetParameter']['matteFePercentage'])
    args.Matte_S_Percentage   = float(req_data['presetParameter']['matteSPercentage'])
    args.Slag_Cu_Percentage   = float(req_data['presetParameter']['slagCuPercentage'])
    args.Slag_Fe_Percentage   = float(req_data['presetParameter']['slagFePercentage'])
    args.Slag_S_Percentage    = float(req_data['presetParameter']['slagSPercentage'])
    args.Slag_SiO2_Percentage = float(req_data['presetParameter']['slagSiO2Percentage'])
    args.MAX_TYPE_TO_SEARCH   = int(req_data['presetParameter']['maxType'])
    args.OXYGEN_PEER_COAL     = float(req_data['presetParameter']['oxygenPeaCoalRatio'])
    args.OXYGEN_CONCENTRATION = float(req_data['presetParameter']['oxygenConcentration'])
    args.COAL_T               = float(req_data['presetParameter']['peaCoal'])
    args.Fe_vs_SiO2           = float(req_data['presetParameter']['FeSiO2Ratio'])
    args.Fe3O4_vs_FeO         = float(req_data['presetParameter']['Fe3O4_vs_FeO'])
    args.Flow                 = float(req_data['presetParameter']['consumedAmount'])
    args.RecallRate           = float(req_data['presetParameter']['recallRate'])
    return args

@app.route('/api/calculate', methods=['POST', 'GET'])
@cross_origin()
def calculate():    #API 1,
    #try:
        req_data = request.get_json()
    
        #req_data to pd:
        pd_data = req_to_pd(req_data['list'])
    
        #Web-set parameters
        global args
        args = get_presets(args, req_data)
        args.INGREDIENT_STORAGE = pd_data   #NOTE 接收的配料基础数据是当前的库存
        args.INGREDIENT_STORAGE = args.INGREDIENT_STORAGE.fillna(0)
    
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
            elements.update({i['name']:[float(i['percentage'])]})
            try:
                priorities.append(float(i['priority']))
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
        _best_ratio_adjust_, _y_, best_solution, element_output = run_opt(args)
        raw_ratio = best_solution['calculatePercentage']
        best_solution = best_solution.loc[best_solution['calculatePercentage']!=0]
        best_solution = web_ratio_int(best_solution)
        best_solution, element_output = mixing(args, best_solution)
        best_solution = web_consumption_int(best_solution)
        _, _y_ = evaluation(args, best_solution, element_output)
    
        #计算氧料比：
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
                "totalLeftOver": round(sum(best_solution['inventoryBalance']), 2),
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
    #except Exception as e:
    #    print('Error', e)
    #    return jsonify({'error': str(e)}), 405

if __name__ == '__main__':
    doc = 'GA搜索和“三种必选（仅必选，必选且用完，必选且比例）”的关系：只有“仅必选”参与GA搜索，同时GA的5%阈值考虑“必选且比例”，原则上“必选且用完”和“必选且比例”在GA外生成solution时候才被加入，evaluation的时候“必选且用完”要另外单独考虑。'
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--DEBUG', action='store_true', default=False)
    parser.add_argument("-S", '--ON_SERVER', action='store_false', default=True)
    parser.add_argument("-O", '--OBJ', type=int, default=1)
    parser.add_argument("-E", '--epoch', type=int, default=26)
    parser.add_argument("-P", '--pop', type=int, default=26)
    parser.add_argument("-A", '--alpha', type=int, default=1)
    parser.add_argument("-B", '--beta', type=int, default=1)
    parser.add_argument("-G", '--gama', type=int, default=1)  #default=3~4  ~=2*alpha+1*beta
    parser.add_argument("-M", '--MAX_TYPE_TO_SEARCH', type=int, default=4)
    parser.add_argument("--NOT_COMPUTE", type=list, default=['渣精矿烟灰', '渣精矿混烟灰', '渣精矿'])
    parser.add_argument('--Flow', type=int, default=150)
    parser.add_argument("-ELEMENTS", '--ELEMENTS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O', 'Sb', 'Bi', 'Ni', 'Ag', 'Au'])
    parser.add_argument("-ELEMENT_MATTERS", '--ELEMENT_MATTERS', type=list, default=['Cu', 'As'])
    parser.add_argument('--OXYGEN_CONCENTRATION', type=float, default=0.85)
    parser.add_argument('--COAL_T', type=float, default=1.5)
    parser.add_argument('--OXYGEN_PEER_COAL', type=float, default=1100)
    parser.add_argument('--Fe_vs_SiO2', type=float, default=2)
    args = parser.parse_args()
    args.ELEMENT_PRIORITIES_SCORE = np.array([1]*len(args.ELEMENT_MATTERS))
    args.ADJUSTED_TOKEN = False
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
        full_solution = SOLUTION
        #full_solution, element_output = mixing(args, full_solution)
        #3.评判标准
        #_, scores = evaluation(args, full_solution, element_output)
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


#目前已经把‘仅必选、必选且用光’两个都传给GA了，‘必备且有百分比’的项目在外面做
#仅必选的通过扩大100倍保留了，必用光的多项有reducer，但貌似没有做惩罚。先这样吧。




