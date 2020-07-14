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
app = Flask(__name__)
import random
import miscs

def compelete_basic_args(args, req_data):
    args.data_all = miscs.req_to_pd(req_data)
    #数据准备与计算部分
    args.data_all['name'] = args.data_all['??]
    args.data_all = args.data_all.set_index('number')
    data = args.data_all[args.NEED_TO_CORRECT + ['material']]
    #In row:
    args.material_in = data[data['material'] == '原料']   #TODO，中间物料还有一部分是输入或in部分。
    #Out row:
    out_type = list(set(data['material'])-{'原料','中间物料'})  #除了原料、中间物料其他都是out
    args.material_out = pd.DataFrame(columns = args.material_in.columns, dtype=np.float)
    for this_type in out_type:
        this_out = data[data['material'] == this_type]   #TODO，中间物料还有一部分是输入或in部分
        args.material_out = pd.concat([args.material_out, this_out])
    #数据直接取出
    #平衡左：
    args.Dry_T_in = args.material_in['currentBalanceDry'].values
    args.Cu_T_in = (args.material_in['currentBalanceDry']*args.material_in['currentBalancePercentageCu']).values  #T
    args.Au_g_in = (args.material_in['currentBalanceDry']*args.material_in['currentBalanceUnitageAu']).values #g
    args.Ag_g_in = (args.material_in['currentBalanceDry']*args.material_in['currentBalanceUnitageAg']).values  #g
    #平衡右：
    args.Dry_T_out = -args.material_out['currentBalanceDry'].values  #Note:平衡右边，物料输出取负值。
    args.Cu_T_out = (args.material_out['currentBalanceDry']*args.material_out['currentBalancePercentageCu']).values  #T
    args.Au_g_out = (args.material_out['currentBalanceDry']*args.material_out['currentBalanceUnitageAu']).values  #g
    args.Ag_g_out = (args.material_out['currentBalanceDry']*args.material_out['currentBalanceUnitageAg']).values  #g
    #全部观测量，无论物料入、物料出、or 含量入、含量出，都是GA的优化维度
    args.obs_T = np.hstack((args.Dry_T_in, args.Dry_T_out))
    args.obs_Cu = np.hstack((args.Cu_T_in, args.Cu_T_out))
    args.obs_Au = np.hstack((args.Au_g_in, args.Au_g_out))
    args.obs_Ag = np.hstack((args.Ag_g_in, args.Ag_g_out))
    #variances:  #TODO read in variances
    args.obs_variance_wrt_T =  (0.1*args.obs_T)**2
    args.obs_variance_wrt_Cu = (0.01*args.obs_Cu)**2
    args.obs_variance_wrt_Au = (0.01*args.obs_Au)**2
    args.obs_variance_wrt_Ag = (0.01*args.obs_Ag)**2
    #bounds:  #TODO, read in bounds
    args.obs_T_bounds = np.array([args.obs_T*0.7, args.obs_T*1.3])
    args.obs_Cu_bounds = np.array([args.obs_Cu*0.97, args.obs_Cu*1.03])
    args.obs_Au_bounds = np.array([args.obs_Au*0.97, args.obs_Au*1.03])
    args.obs_Ag_bounds = np.array([args.obs_Ag*0.97, args.obs_Ag*1.03])
    #Note: 注意边界的顺序是：吨、铜、金、银
    args.lower_bounds = np.hstack((args.obs_T_bounds[0,:], args.obs_Cu_bounds[0,:], args.obs_Au_bounds[0,:], args.obs_Ag_bounds[0,:]))
    args.upper_bounds = np.hstack((args.obs_T_bounds[1,:], args.obs_Cu_bounds[1,:], args.obs_Au_bounds[1,:], args.obs_Ag_bounds[1,:]))
    #如有必要，交换上下界
    switch_index = np.where(args.lower_bounds > args.upper_bounds)
    tmp_lower = copy.deepcopy(args.lower_bounds[switch_index])
    args.lower_bounds[switch_index] = copy.deepcopy(args.upper_bounds[switch_index])
    args.upper_bounds[switch_index] = tmp_lower

    #GA basic:
    args.NUM_OF_TYPES_FOR_GA = len(args.obs_T) + len(args.obs_Cu) + len(args.obs_Au) + len(args.obs_Ag)
    #args.precisions = 1 / (10**(np.array([miscs.scale_and_precision(i)[1] for i in args.lower_bounds])+1))
    args.precisions = 0.1*np.ones(shape=args.lower_bounds.shape)
    return args

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

def GAwrapper(ga_outcomes):   #ga_outcomes是遗传算法给过来的,是需要优化得到的各种真实值:ground_truth.
    global args
    #取出不同部分，用于分别计算目标函数：
    ga_outcomes = ga_outcomes.reshape(-1, args.NUM_OF_TYPES_FOR_GA)
    ga_T = ga_outcomes[:, :len(args.obs_T)]
    ga_Cu = ga_outcomes[:, len(args.obs_T):len(args.obs_T)+len(args.obs_Cu)]
    ga_Au = ga_outcomes[:, len(args.obs_T)+len(args.obs_Cu):len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au)]
    ga_Ag = ga_outcomes[:, len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au):]
    #评价函数
    scores = evaluation(ga_T, ga_Cu, ga_Au, ga_Ag)
    return scores

def evaluation(ga_T, ga_Cu, ga_Au, ga_Ag):
    #Evaluations:
    T_part = ((args.obs_T - ga_T)**2 / args.obs_variance_wrt_T).sum(axis=1)  
    Cu_part = ((args.obs_Cu - ga_Cu)**2 / args.obs_variance_wrt_Cu).sum(axis=1)  
    Au_part = ((args.obs_Au - ga_Au)**2 / args.obs_variance_wrt_Au).sum(axis=1)  
    Ag_part = ((args.obs_Ag - ga_Ag)**2 / args.obs_variance_wrt_Ag).sum(axis=1)  
    #目标函数是最大似然的幂指数加和:
    #GA适应度需要最大值，但GA自己取了负数，所以幂次直接求最小值即可，不用任何转换
    scores = args.WEIGHT_T_VOLUME*T_part + args.WEIGHT_CU_PERCENTAGE*Cu_part + args.WEIGHT_AU_PERCENTAGE*Au_part + args.WEIGHT_AG_PERCENTAGE*Ag_part

    #Penalties as scores:
    T_in = ga_T[:,:len(args.material_in)]
    T_out = ga_T[:,len(args.material_in):]
    Cu_in = ga_Cu[:,:len(args.material_in)]
    Cu_out = ga_Cu[:,len(args.material_in):]
    Au_in = ga_Au[:,:len(args.material_in)]
    Au_out = ga_Au[:,len(args.material_in):]
    Ag_in = ga_Ag[:,:len(args.material_in)]
    Ag_out = ga_Ag[:,len(args.material_in):]
    Cu_balance = np.abs((T_in * Cu_in).sum(axis=1) + (T_out * Cu_out).sum(axis=1))
    Au_balance = np.abs((T_in * Au_in).sum(axis=1) + (T_out * Au_out).sum(axis=1))
    Ag_balance = np.abs((T_in * Ag_in).sum(axis=1) + (T_out * Ag_out).sum(axis=1))
    scores += args.WEIGHT_BALANCE * (Cu_balance + Au_balance + Ag_balance)
    if args.IS_VECTOR:
        scores = scores
    else:
        scores = scores[0]
    return scores

def run_opt(args):
    print("Single thread (Always in metal balancing)... Vector mode: %s"%args.IS_VECTOR)
    #for attr in dir(args):
    #    if not attr.startswith('_'):print(attr)
    constraint_eq, constraint_ueq = get_constraints(args)
    GAwrapper.is_vector = args.IS_VECTOR
    #考虑一步到位,所有物料参与选择,下限为0
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.POP, max_iter=args.EPOCH, lb=args.lower_bounds, ub=args.upper_bounds, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=args.precisions, prob_mut=0.01)
    best_gax, best_gay = ga.run()
    if args.PLOT:
        import matplotlib.pyplot as plt
        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()
    return best_gax, best_gay

@app.route('/api/correct_data', methods=['POST', 'GET'])
@cross_origin()
def correct_data():    #API 
    #输入部分
    req_data = request.get_json()
    req_data = req_data['list']
    global args
    args = compelete_basic_args(args, req_data)

    #优化部分
    best_x, best_y = run_opt(args) 
    #从各自的既定位置取出
    best_x = best_x.reshape(-1, args.NUM_OF_TYPES_FOR_GA)
    ga_T = best_x[:, :len(args.obs_T)]
    ga_Cu = best_x[:, len(args.obs_T):len(args.obs_T)+len(args.obs_Cu)]
    ga_Au = best_x[:, len(args.obs_T)+len(args.obs_Cu):len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au)]
    ga_Ag = best_x[:, len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au):]
    #放回要给web的列表（先前输出量有取成负值，此处为方便直接整个取abs，即可获得正确结果）
    row_names = list(args.material_in.index) + list(args.material_out.index)
    args.data_all.loc[row_names, 'calibrated_currentBalanceDry'] = np.abs(np.round(ga_T.flatten()))
    args.data_all.loc[row_names, 'calibrated_currentBalancePercentageCu'] = np.abs(np.round(ga_Cu.flatten(), 3))
    args.data_all.loc[row_names, 'calibrated_currentBalanceUnitageAu'] = np.abs(np.round(ga_Au.flatten(), 2))
    args.data_all.loc[row_names, 'calibrated_currentBalanceUnitageAg'] = np.abs(np.round(ga_Ag.flatten(), 2))
    args.data_all.loc[row_names, 'currentBalanceDry'] = args.data_all.loc[row_names, 'calibrated_currentBalanceDry']
    args.data_all.loc[row_names, 'currentBalancePercentageCu'] = args.data_all.loc[row_names, 'calibrated_currentBalancePercentageCu']
    args.data_all.loc[row_names, 'currentBalanceUnitageAu'] = args.data_all.loc[row_names, 'calibrated_currentBalanceUnitageAu']
    args.data_all.loc[row_names, 'currentBalanceUnitageAg'] = args.data_all.loc[row_names, 'calibrated_currentBalanceUnitageAg']
    args.data_all = args.data_all.fillna(0)
    args.data_all['number'] = args.data_all.index
    #回收率
    #铜回收率%=本月产出阴极铜、电积铜÷（本期使用原料+上月中间结存-本月中间结存-阳极泥）×100；
    #银回收率%=本月产出阳极泥÷（本期使用原料+上月中间结存-本月中间结存）×100；
    #金回收率%=本月产出阳极泥÷（本期使用原料+上月中间结存-本月中间结存）×100
    r_Ag = args.data_all['recoveryAg'].values[np.where(args.data_all['recoveryAg']>0)[0]][0]
    r_Au = args.data_all['recoveryAu'].values[np.where(args.data_all['recoveryAu']>0)[0]][0]
    r_Cu = args.data_all['recoveryCu'].values[np.where(args.data_all['recoveryCu']>0)[0]][0]
    #返回部分
    res_data = miscs.pd_to_res(args.data_all)
    res_data = {
           'list':res_data,
           'parameter':{'recoveryAg':r_Ag, 'recoveryAu':r_Au, 'recoveryCu':r_Cu}
            }
    embed()
    return jsonify(res_data)


if __name__ == '__main__':
    doc = '金属平衡需要解决‘什么样的真实值最优可能获得目前的观测值’的最大似然问题～求解过程见doc文档，此处从目标函数开始编程。'
    parser = argparse.ArgumentParser()
    parser.add_argument('--COAL_T', type=float, default=1.5)
    parser.add_argument("-E", '--EPOCH', type=int, default=100)
    parser.add_argument("-P", '--POP', type=int, default=5000)
    parser.add_argument('--WEIGHT_T_VOLUME', type=int, default=1)   #volume (T)
    parser.add_argument("--WEIGHT_CU_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_AU_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_AG_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_BALANCE", type=int, default=1) # for Cu, Au, Ag
    parser.add_argument("-M", '--MAX_TYPE_TO_SEARCH', type=int, default=10)
    parser.add_argument("-V", '--IS_VECTOR', action='store_true', default=False)
    parser.add_argument('--PLOT', action='store_true', default=False)
    parser.add_argument("--NEED_TO_CORRECT", type=list, default=['currentBalanceDry','currentBalancePercentageCu','currentBalanceUnitageAg','currentBalanceUnitageAu'])

    args = parser.parse_args()

    manager = Manager()
    normed_dict = manager.dict()
    normed_dict['normed_obj_amount'] = manager.list()

    app.run(host='0.0.0.0', port=7001, debug=True)



