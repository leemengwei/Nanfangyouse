#!/usr/bin/python
import argparse
import pandas as pd
import numpy as np
import os,sys,time
from IPython import embed
import matplotlib.pyplot as plt
from sko.GA import GA 
import numpy as np
import copy
#NETWORK
from flask import Flask, request, jsonify
from flask_cors import cross_origin  #戚总-张驰API
app = Flask(__name__)
import miscs

COLS = ['lastBalanceDry', 'lastBalancePercentageCu', 'lastBalanceUnitageAg', 'lastBalanceUnitageAu', 'lastBalanceCu', 'lastBalanceAu', 'lastBalanceAg', \
        'currentIncomeDry', 'currentIncomeCu', 'currentIncomeAu', 'currentIncomeAg', 'currentIncomePercentageCu', 'currentIncomeUnitageAg', 'currentIncomeUnitageAu', \
        'currentBalanceDry', 'currentBalancePercentageCu', 'currentBalanceUnitageAg', 'currentBalanceUnitageAu', 'currentBalanceCu', 'currentBalanceAu', 'currentBalanceAg', \
        'currentCostAg', 'currentCostAu', 'currentCostCu', 'currentCostDry']

def compelete_basic_args(args, req, by_correct_data=False):
    args.AUTO_WEIGHTS = {}
    req_data = req['list']
    args.factory = req['factory']
    args.data_all = miscs.req_to_pd(req_data)
    #数据准备与计算部分
    args.data_all['name'] = 'token'.join(list(args.data_all.index)).replace(' ', '').split('token')
    args.data_all = args.data_all.fillna(0)
    args.data_all.loc[args.data_all['material']=='中间产品','material'] = '产品'  #NOTE: 都统一写为产品，要用做计算直收、回收率
    data = args.data_all
    #不再区分输入输出：------------------------------------------------------
    #####In row:
    ####args.material_in = data[data['material'] == '输入']   #中间物料还有一部分是输入或in部分。
    #####Out row:
    ####args.material_out = data[data['material'] == '输出']   #中间物料还有一部分是输入或in部分。
    #####Out row:
    #####out_type = list(set(data['material'])-{'原料'})  #除了原料其他都是out?
    #####args.material_out = pd.DataFrame(columns = args.material_in.columns, dtype=np.float)
    #####for this_type in out_type:
    #####    this_out = data[data['material'] == this_type] 
    #####    args.material_out = pd.concat([args.material_out, this_out])
    ####old 根据之前指定的material in和out，out取负值，随后用它给定ga上下限
    ####NOTE: 没有所谓的in和out，表中数据计算到消耗后，就应该是自平衡的，直接sum就应该为0，输出项的量也不再手动取负。此处仅仅取出一些值，用于evaluate时进行平衡计算。
    #上期盘点：
    args.lastBalanceDry             = data['lastBalanceDry'].values
    args.lastBalancePercentageCu    = data['lastBalancePercentageCu'].values
    #args.lastBalanceCu              = data['lastBalanceCu'].values
    args.lastBalanceUnitageAu       = data['lastBalanceUnitageAu'].values
    #args.lastBalanceAu              = data['lastBalanceAu'].values
    args.lastBalanceUnitageAg       = data['lastBalanceUnitageAg'].values
    #args.lastBalanceAg              = data['lastBalanceAg'].values
    #本期投入：
    args.currentIncomeDry           = data['currentIncomeDry'].values
    args.currentIncomePercentageCu  = data['currentIncomePercentageCu'].values
    #args.currentIncomeCu            = data['currentIncomeCu'].values
    args.currentIncomeUnitageAg     = data['currentIncomeUnitageAg'].values
    #args.currentIncomeAg            = data['currentIncomeAg'].values
    args.currentIncomeUnitageAu     = data['currentIncomeUnitageAu'].values
    #args.currentIncomeAu            = data['currentIncomeAu'].values
    #本期盘点：
    args.currentBalanceDry          = data['currentBalanceDry'].values
    args.currentBalancePercentageCu = data['currentBalancePercentageCu'].values
    #args.currentBalanceCu           = data['currentBalanceCu'].values
    args.currentBalanceUnitageAg    = data['currentBalanceUnitageAg'].values
    #args.currentBalanceAg           = data['currentBalanceAg'].values
    args.currentBalanceUnitageAu    = data['currentBalanceUnitageAu'].values
    #args.currentBalanceAu           = data['currentBalanceAu'].values

    #当期库存的全部观测量，无论出、入,之后都会做GA的优化维度
    #args.obs_T = np.hstack((args.material_in['currentBalanceDry'].values, args.material_out['currentBalanceDry'].values))  
    #args.obs_Cu = np.hstack((args.material_in['currentBalancePercentageCu'].values, args.material_out['currentBalancePercentageCu'].values))
    #args.obs_Au = np.hstack((args.material_in['currentBalanceUnitageAu'].values, args.material_out['currentBalanceUnitageAu'].values))
    #args.obs_Ag = np.hstack((args.material_in['currentBalanceUnitageAg'].values, args.material_out['currentBalanceUnitageAg'].values))
    args.obs_T = args.currentBalanceDry
    args.obs_Cu = args.currentBalancePercentageCu
    args.obs_Au = args.currentBalanceUnitageAu
    args.obs_Ag = args.currentBalanceUnitageAg

    if by_correct_data:
        req_settings = req['setting']
        args.settings = miscs.req_to_pd(req_settings)
        #方差variances: 
        args.settings.currentBalanceDryVariance = args.settings.currentBalanceDryVariance.astype(float)
        args.settings.currentBalancePercentageCuVariance = args.settings.currentBalancePercentageCuVariance.astype(float)
        args.settings.currentBalanceUnitageAuVariance = args.settings.currentBalanceUnitageAuVariance.astype(float)
        args.settings.currentBalanceUnitageAgVariance = args.settings.currentBalanceUnitageAgVariance.astype(float)
        args.obs_variance_wrt_T  = (args.settings.currentBalanceDryVariance.values)**2 + epsilon
        args.obs_variance_wrt_Cu = (args.settings.currentBalancePercentageCuVariance.values)**2 + epsilon
        args.obs_variance_wrt_Au = (args.settings.currentBalanceUnitageAuVariance.values)**2 + epsilon
        args.obs_variance_wrt_Ag = (args.settings.currentBalanceUnitageAgVariance.values)**2 + epsilon
        #上下限bounds:
        args.settings.currentBalancePercentageCuMin = args.settings.currentBalancePercentageCuMin.astype(float)
        args.settings.currentBalancePercentageCuMax = args.settings.currentBalancePercentageCuMax.astype(float)
        args.settings.currentBalanceDryMin = args.settings.currentBalanceDryMin.astype(float)
        args.settings.currentBalanceDryMax = args.settings.currentBalanceDryMax.astype(float)
        args.settings.currentBalanceUnitageAuMin = args.settings.currentBalanceUnitageAuMin.astype(float)
        args.settings.currentBalanceUnitageAuMax = args.settings.currentBalanceUnitageAuMax.astype(float)
        args.settings.currentBalanceUnitageAgMin = args.settings.currentBalanceUnitageAgMin.astype(float)
        args.settings.currentBalanceUnitageAgMax = args.settings.currentBalanceUnitageAgMax.astype(float)
        #NOTE: 铜的需要clip一下最大不超过99.5%
        args.settings.currentBalancePercentageCuMin = np.clip(args.settings.currentBalancePercentageCuMin.values, 0, 99.5)
        args.settings.currentBalancePercentageCuMax = np.clip(args.settings.currentBalancePercentageCuMax.values, 0, 99.5)
        args.obs_T_bounds = np.array([args.settings.currentBalanceDryMin.values, args.settings.currentBalanceDryMax.values+epsilon])
        args.obs_Cu_bounds = np.array([args.settings.currentBalancePercentageCuMin.values, args.settings.currentBalancePercentageCuMax.values+epsilon])
        args.obs_Au_bounds = np.array([args.settings.currentBalanceUnitageAuMin.values, args.settings.currentBalanceUnitageAuMax.values+epsilon])
        args.obs_Ag_bounds = np.array([args.settings.currentBalanceUnitageAgMin.values, args.settings.currentBalanceUnitageAgMax.values+epsilon])
        #NOTE: 注意边界的顺序是：吨、铜、金、银
        args.lower_bounds = np.hstack((args.obs_T_bounds[0,:], args.obs_Cu_bounds[0,:], args.obs_Au_bounds[0,:], args.obs_Ag_bounds[0,:]))
        args.upper_bounds = np.hstack((args.obs_T_bounds[1,:], args.obs_Cu_bounds[1,:], args.obs_Au_bounds[1,:], args.obs_Ag_bounds[1,:]))
        #如有必要，交换上下界
        switch_index = np.where(args.lower_bounds > args.upper_bounds)
        tmp_lower = copy.deepcopy(args.lower_bounds[switch_index])
        args.lower_bounds[switch_index] = copy.deepcopy(args.upper_bounds[switch_index])
        args.upper_bounds[switch_index] = tmp_lower
        args.upper_bounds += epsilon   #Add tiny to aviod nan in GA
        #GA basic:
        args.NUM_OF_TYPES_FOR_GA = len(args.obs_T) + len(args.obs_Cu) + len(args.obs_Au) + len(args.obs_Ag)
        #args.precisions = 1 / (10**(np.array([miscs.scale_and_precision(i)[1] for i in args.upper_bounds])+1))
        #NOTE: precision & epsilon 的关系，否则会导致ga生成nan
        args.precisions = 1e-5
    else:
        pass
    return args

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
    #time.sleep(0.1)
    print(scores.min())
    return scores

def evaluation(ga_T, ga_Cu, ga_Au, ga_Ag):
    global args
    #Evaluations:
    T_prob  = ((args.obs_T - ga_T)**2 / (args.obs_variance_wrt_T+epsilon)).sum(axis=1)  
    Cu_prob = ((args.obs_Cu - ga_Cu)**2 / (args.obs_variance_wrt_Cu+epsilon)).sum(axis=1)  
    Au_prob = ((args.obs_Au - ga_Au)**2 / (args.obs_variance_wrt_Au+epsilon)).sum(axis=1)  
    Ag_prob = ((args.obs_Ag - ga_Ag)**2 / (args.obs_variance_wrt_Ag+epsilon)).sum(axis=1)  
    #Penalties as scores:
    #T_in = ga_T[:,:len(args.material_in)]
    #T_out = ga_T[:,len(args.material_in):]
    #Cu_in = ga_Cu[:,:len(args.material_in)]
    #Cu_out = ga_Cu[:,len(args.material_in):]
    #Au_in = ga_Au[:,:len(args.material_in)]
    #Au_out = ga_Au[:,len(args.material_in):]
    #Ag_in = ga_Ag[:,:len(args.material_in)]
    #Ag_out = ga_Ag[:,len(args.material_in):]
    #各个平衡的表达如下：out已经取得负值，所以加和为零（求最小）即为平衡
    #Cu_balance = np.abs((T_in * Cu_in).sum(axis=1) + (T_out * Cu_out).sum(axis=1))
    #Au_balance = np.abs((T_in * Au_in).sum(axis=1) + (T_out * Au_out).sum(axis=1))
    #Ag_balance = np.abs((T_in * Ag_in).sum(axis=1) + (T_out * Ag_out).sum(axis=1))
    #NOTE:各个平衡的新表达如下：本期使用每个元素维度加和都为零（求最小）即为平衡
    Cu_balance = np.abs((args.lastBalanceDry*args.lastBalancePercentageCu + args.currentIncomeDry*args.currentIncomePercentageCu - ga_T*ga_Cu).sum(axis=1))/100   #% --> /100, t
    Au_balance = np.abs((args.lastBalanceDry*args.lastBalanceUnitageAu + args.currentIncomeDry*args.currentIncomeUnitageAu - ga_T*ga_Au).sum(axis=1))/1000        #g, kg -->/1000
    Ag_balance = np.abs((args.lastBalanceDry*args.lastBalanceUnitageAg + args.currentIncomeDry*args.currentIncomeUnitageAg - ga_T*ga_Ag).sum(axis=1))/1000        #g, kg -->/1000
    #单EPOCH， 大POP，使用均值MEAN作为自平衡系数（分母）：
    if args.AUTO_WEIGHTS == {}:
        args.AUTO_WEIGHTS['T_prob_weights'] = T_prob.std()
        args.AUTO_WEIGHTS['Cu_prob_weights'] = Cu_prob.std()
        args.AUTO_WEIGHTS['Au_prob_weights'] = Au_prob.std()
        args.AUTO_WEIGHTS['Ag_prob_weights'] = Ag_prob.std()
        args.AUTO_WEIGHTS['Cu_balance_weights'] = Cu_balance.std()
        args.AUTO_WEIGHTS['Au_balance_weights'] = Au_balance.std()
        args.AUTO_WEIGHTS['Ag_balance_weights'] = Ag_balance.std()
        print("Auto weights generated:", args.AUTO_WEIGHTS)

    #GA适应度需要最大值，但GA自己取了负数，所以幂次直接求最小值即可，不用任何转换
    #1:
    #第一部分的目标函数，是最大似然的幂指数加和:
    scores_mle = args.WEIGHT_T_VOLUME*T_prob/args.AUTO_WEIGHTS['T_prob_weights'] + args.WEIGHT_CU_PERCENTAGE*Cu_prob/args.AUTO_WEIGHTS['Cu_prob_weights'] + args.WEIGHT_AU_PERCENTAGE*Au_prob/args.AUTO_WEIGHTS['Au_prob_weights'] + args.WEIGHT_AG_PERCENTAGE*Ag_prob/args.AUTO_WEIGHTS['Ag_prob_weights']
    #第二部分的目标函数，是平衡约束，越平衡则该值越小，也不用转换：
    scores_balance = args.WEIGHT_BALANCE * (Cu_balance/args.AUTO_WEIGHTS['Cu_balance_weights'] + Au_balance/args.AUTO_WEIGHTS['Au_balance_weights'] + Ag_balance/args.AUTO_WEIGHTS['Ag_balance_weights'])  #NOTE: 铜金银目前非常不平衡，金银的单位是kg，数值上比重此处不平衡
    #2:
    #Ignore weights
    scores_mle = T_prob + Cu_prob + Au_prob + Ag_prob
    scores_balance = Cu_balance + Au_balance + Ag_balance
    #3:
    #mannual
    scores_mle = T_prob/args.AUTO_WEIGHTS['T_prob_weights'] + args.WEIGHT_CU_PERCENTAGE*Cu_prob/args.AUTO_WEIGHTS['Cu_prob_weights'] + args.WEIGHT_AU_PERCENTAGE*Au_prob/args.AUTO_WEIGHTS['Au_prob_weights'] + args.WEIGHT_AG_PERCENTAGE*Ag_prob/args.AUTO_WEIGHTS['Ag_prob_weights']
    #第二部分的目标函数，是平衡约束，越平衡则该值越小，也不用转换：
    scores_balance = 2*Cu_balance/args.AUTO_WEIGHTS['Cu_balance_weights'] + Au_balance/args.AUTO_WEIGHTS['Au_balance_weights'] + Ag_balance/args.AUTO_WEIGHTS['Ag_balance_weights']  #NOTE: 铜金银目前非常不平衡，金银的单位是kg，数值上比重此处不平衡
    #print("T_prob, Cu_prob, Au_prob, Ag_prob, Cu_balance, Au_balance, Ag_balance", T_prob.std(), Cu_prob.std(), Au_prob.std(), Ag_prob.std(), Cu_balance.std(), Au_balance.std(), Ag_balance.std())
    scores = scores_mle + scores_balance
    if args.IS_VECTOR:
        scores = scores
    else:
        scores = scores[0]
    #record:
    args.T_prob_history.append(T_prob) 
    args.Cu_prob_history.append(Cu_prob) 
    args.Au_prob_history.append(Au_prob) 
    args.Ag_prob_history.append(Ag_prob) 
    args.Cu_balance_history.append(Cu_balance)
    args.Au_balance_history.append(Au_balance)
    args.Ag_balance_history.append(Ag_balance)
    return scores

def run_opt(args):
    print("Single thread (Always in metal balancing)... Vector mode: %s"%args.IS_VECTOR)
    #for attr in dir(args):
    #    if not attr.startswith('_'):print(attr)
    GAwrapper.is_vector = args.IS_VECTOR
    #考虑一步到位,所有物料参与选择,下限为0
    #首先获取AUTO_WEIGHTS
    ga_init = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=100000, max_iter=1, lb=args.lower_bounds, ub=args.upper_bounds, precision=args.precisions, prob_mut=0.005)
    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.POP, max_iter=args.EPOCH, lb=args.lower_bounds, ub=args.upper_bounds, precision=args.precisions, prob_mut=0.005)
    best_gax, best_gay = ga.run()
    if args.PLOT:
        import matplotlib.pyplot as plt
        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.title("Y")

        Y_history = pd.DataFrame(args.Cu_balance_history[:-1])
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.title("Cu balance")

        Y_history = pd.DataFrame(args.Au_balance_history[:-1])
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.title("Au balance")

        Y_history = pd.DataFrame(args.Ag_balance_history[:-1])
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.title("Ag balance")

        plt.show()
    return best_gax, best_gay
    
def quick_compute(args):
    args.data_all = args.data_all.fillna(0)
    #上期结存干量:
    args.data_all['lastBalanceCu'] = args.data_all['lastBalancePercentageCu'] * args.data_all['lastBalanceDry']/100
    args.data_all['lastBalanceAu'] = args.data_all['lastBalanceUnitageAu'] * args.data_all['lastBalanceDry']/1000
    args.data_all['lastBalanceAg'] = args.data_all['lastBalanceUnitageAg'] * args.data_all['lastBalanceDry']/1000
    #本期收入干量:
    args.data_all['currentIncomeCu'] = args.data_all['currentIncomePercentageCu'] * args.data_all['currentIncomeDry']/100
    args.data_all['currentIncomeAu'] = args.data_all['currentIncomeUnitageAu'] * args.data_all['currentIncomeDry']/1000
    args.data_all['currentIncomeAg'] = args.data_all['currentIncomeUnitageAg'] * args.data_all['currentIncomeDry']/1000
    #本期结存干量:
    args.data_all['currentBalanceCu'] = args.data_all['currentBalancePercentageCu'] * args.data_all['currentBalanceDry']/100
    args.data_all['currentBalanceAu'] = args.data_all['currentBalanceUnitageAu'] * args.data_all['currentBalanceDry']/1000
    args.data_all['currentBalanceAg'] = args.data_all['currentBalanceUnitageAg'] * args.data_all['currentBalanceDry']/1000
    #本期使用Cost也相应更改
    args.data_all['currentCostDry'] = args.data_all['lastBalanceDry'] + args.data_all['currentIncomeDry'] - args.data_all['currentBalanceDry']
    args.data_all['currentCostCu'] = args.data_all['lastBalanceDry']*args.data_all['lastBalancePercentageCu']/100 + args.data_all['currentIncomeDry']*args.data_all['currentIncomePercentageCu']/100 - args.data_all['currentBalanceDry']*args.data_all['currentBalancePercentageCu']/100  #t
    args.data_all['currentCostAg'] = (args.data_all['lastBalanceDry']*args.data_all['lastBalanceUnitageAg'] + args.data_all['currentIncomeDry']*args.data_all['currentIncomeUnitageAg'] - args.data_all['currentBalanceDry']*args.data_all['currentBalanceUnitageAg'])/1000  #kg
    args.data_all['currentCostAu'] = (args.data_all['lastBalanceDry']*args.data_all['lastBalanceUnitageAu'] + args.data_all['currentIncomeDry']*args.data_all['currentIncomeUnitageAu'] - args.data_all['currentBalanceDry']*args.data_all['currentBalanceUnitageAu'])/1000  #kg

    Cu_balance = sum(args.data_all['currentCostCu'])
    Au_balance = sum(args.data_all['currentCostAu'])
    Ag_balance = sum(args.data_all['currentCostAg'])
    #args.data_all['number'] = args.data_all.index
    #recovery_Cu = np.abs((args.data_all[args.data_all.material!='原料']['currentCostCu'].values.sum()-args.data_all[args.data_all.material=='损失']['currentCostCu'].values.sum())/(args.data_all[args.data_all.material=='原料']['currentCostCu'].values.sum()))*100
    #recovery_Au = np.abs((args.data_all[args.data_all.material!='原料']['currentCostAu'].values.sum()-args.data_all[args.data_all.material=='损失']['currentCostAu'].values.sum())/(args.data_all[args.data_all.material=='原料']['currentCostAu'].values.sum()))*100
    #recovery_Ag = np.abs((args.data_all[args.data_all.material!='原料']['currentCostAg'].values.sum()-args.data_all[args.data_all.material=='损失']['currentCostAg'].values.sum())/(args.data_all[args.data_all.material=='原料']['currentCostAg'].values.sum()))*100
    #回收率recovery是指回收的产品和还有回收价值的渣类、烟尘等(仅仅除去废料)与投入的原料比值。
    #直收率recall是直接回收出来的产品和投入原料的比。
    if args.factory == '熔炼厂':  #熔炼厂：
        #铜、银、金回收率%=本期产出阳极铜÷（本期使用原料+前期中间结存-本期中间结存-本期产出熔炼渣）×100
        recovery_Cu = np.abs(args.data_all[args.data_all.name=='阳极铜']['currentCostCu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostCu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostCu'].sum() - args.data_all[args.data_all.name=='熔炼渣']['currentCostCu'].sum()))*100
        recovery_Au = np.abs(args.data_all[args.data_all.name=='阳极铜']['currentCostAu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAu'].sum() - args.data_all[args.data_all.name=='熔炼渣']['currentCostAu'].sum()))*100
        recovery_Ag = np.abs(args.data_all[args.data_all.name=='阳极铜']['currentCostAg'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAg'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAg'].sum() - args.data_all[args.data_all.name=='熔炼渣']['currentCostAg'].sum()))*100
    elif args.factory == '精炼厂':
        #铜回收率%=本期产出阴极铜÷（本期使用原料+前期中间结存-本期中间结存-本期产出回收品）×100    #TODO:精炼厂：阴极铜没有啊
        #银回收率%=本期产出阳极泥÷（本期使用原料+前期中间结存-本期中间结存-本期产出残极板-本期产出铜屑）×100
        #金回收率%=本期产出阳极泥÷（本期使用原料+前期中间结存-本期中间结存-本期产出残极板-本期产出铜屑）×100
        recovery_Cu = np.abs(args.data_all[args.data_all.name=='阴极铜']['currentCostCu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostCu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostCu'].sum() - args.data_all[args.data_all.material=='回收品']['currentCostCu'].sum()))*100
        recovery_Au = np.abs(args.data_all[args.data_all.name=='阳极泥']['currentCostAu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAu'].sum() - args.data_all[args.data_all.name=='残极板']['currentCostAu'].sum() - args.data_all[args.data_all.name=='铜屑']['currentCostAu'].sum()))*100
        recovery_Ag = np.abs(args.data_all[args.data_all.name=='阳极泥']['currentCostAg'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAg'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAg'].sum() - args.data_all[args.data_all.name=='残极板']['currentCostAg'].sum() - args.data_all[args.data_all.name=='铜屑']['currentCostAg'].sum()))*100
    elif args.factory == '综合厂':
        #铜、金、银回收率%=本期产出渣精矿÷本期使用原料×100
        recovery_Cu = np.abs(args.data_all[args.data_all.name=='渣精矿']['currentCostCu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostCu'].sum()))*100
        recovery_Au = np.abs(args.data_all[args.data_all.name=='渣精矿']['currentCostAu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAu'].sum()))*100
        recovery_Ag = np.abs(args.data_all[args.data_all.name=='渣精矿']['currentCostAg'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAg'].sum()))*100
    elif args.factory == '全厂':
        #铜回收率%=本月产出阴极铜、电积铜÷（本期使用原料+上月中间结存-本月中间结存-阳极泥）×100    #TODO:包括旋流？
        #银回收率%=本月产出阳极泥÷（本期使用原料+上月中间结存-本月中间结存）×100
        #金回收率%=本月产出阳极泥÷（本期使用原料+上月中间结存-本月中间结存）×100
        recovery_Cu = np.abs((args.data_all[args.data_all.name=='阴极铜']['currentCostCu'].sum() + args.data_all[args.data_all.name=='电积铜']['currentCostCu'].sum()) / (args.data_all[args.data_all.material=='原料']['currentCostCu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostCu'].sum() - args.data_all[args.data_all.name=='阳极泥']['currentCostCu'].sum()))*100
        recovery_Au = np.abs(args.data_all[args.data_all.name=='阳极泥']['currentCostAu'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAu'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAu'].sum()))*100
        recovery_Ag = np.abs(args.data_all[args.data_all.name=='阳极泥']['currentCostAg'].sum() / (args.data_all[args.data_all.material=='原料']['currentCostAg'].sum() + args.data_all[args.data_all.material=='中间物料']['currentCostAg'].sum()))*100
    else:
        print("Error factory:", args.factory)
    #Cu直收率=产品含Cu÷（本期使用原料含铜+本期使用中间结存含Cu-本期产出熔炼渣含Cu-各种废？）×100
    #Ag直收率=产品含Ag÷（本期使用原料含Ag+本期使用中间结存含Ag-本期产出熔炼渣含Ag-各种废？）×100
    #Au直收率=产品含Au÷（本期使用原料含Au+本期使用中间结存含Au-本期产出熔炼渣含Au-各种废？）×100
    recall_Cu = np.abs(args.data_all[args.data_all.material=='产品']['currentCostCu'].values.sum()/(args.data_all[args.data_all.material=='原料']['currentCostCu'].values.sum()))*100
    recall_Au = np.abs(args.data_all[args.data_all.material=='产品']['currentCostAu'].values.sum()/(args.data_all[args.data_all.material=='原料']['currentCostAu'].values.sum()))*100
    recall_Ag = np.abs(args.data_all[args.data_all.material=='产品']['currentCostAg'].values.sum()/(args.data_all[args.data_all.material=='原料']['currentCostAg'].values.sum()))*100

    #round up1:
    for col in COLS:
        args.data_all[col] = np.round(args.data_all[col], 5)
    #round up2:
    recovery_Ag = np.round(recovery_Ag, 5)
    recovery_Au = np.round(recovery_Au, 5)
    recovery_Cu = np.round(recovery_Cu, 5)
    recall_Ag = np.round(recall_Ag, 5)
    recall_Au = np.round(recall_Au, 5)
    recall_Cu = np.round(recall_Cu, 5)
    Cu_balance = np.round(Cu_balance, 5)
    Au_balance = np.round(Au_balance, 5)
    Ag_balance = np.round(Ag_balance, 5)
    return args, recovery_Ag, recovery_Au, recovery_Cu, recall_Ag, recall_Au, recall_Cu, Cu_balance, Au_balance, Ag_balance

@app.route('/api/quick_update',methods=['POST', 'GET'])
@cross_origin()
def quick_update():    #API 
    try:
        #输入部分
        req = request.get_json()
        global args
        args = compelete_basic_args(args, req, by_correct_data=False)
    
        #计算消耗、全场回收率
        args, recovery_Ag, recovery_Au, recovery_Cu, recall_Ag, recall_Au, recall_Cu, Cu_balance, Au_balance, Ag_balance = quick_compute(args)
    
        #返回部分
        res_data = miscs.pd_to_res(args.data_all)
        res_data = {
               'list':res_data,
               'parameter':{'recoveryAg':recovery_Ag, \
                            'recoveryAu':recovery_Au, \
                            'recoveryCu':recovery_Cu, \
                            'recallAg':recall_Ag, \
                            'recallAu':recall_Au, \
                            'recallCu':recall_Cu, \
                            'Ag_balance':Ag_balance, \
                            'Au_balance':Au_balance, \
                            'Cu_balance':Cu_balance},
               'set': [0]
                }
        return jsonify(res_data)
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405

@app.route('/api/overall',methods=['POST', 'GET'])
@cross_origin()
def overall():    #API 
    try:
        req = request.get_json()
        req['list'] = req['rlResizeList'] + req['jlResizeList'] + req['zhResizeList']
        global args
        args = compelete_basic_args(args, req, by_correct_data=False)
        args, recovery_Ag, recovery_Au, recovery_Cu, recall_Ag, recall_Au, recall_Cu, Cu_balance, Au_balance, Ag_balance = quick_compute(args)
        res_data = miscs.pd_to_res(args.data_all)
        res_data = {
               'list':res_data,
               'parameter':{'recoveryAg':recovery_Ag, \
                            'recoveryAu':recovery_Au, \
                            'recoveryCu':recovery_Cu, \
                            'recallAg':recall_Ag, \
                            'recallAu':recall_Au, \
                            'recallCu':recall_Cu, \
                            'Ag_balance':Ag_balance, \
                            'Au_balance':Au_balance, \
                            'Cu_balance':Cu_balance},
               'set': [0]
                }
        return jsonify(res_data)
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405

@app.route('/api/correct_data',methods=['POST', 'GET'])
@cross_origin()
def correct_data():    #API 
    try:
        #输入部分
        req = request.get_json()
        global args
        args = compelete_basic_args(args, req, by_correct_data=True)
    
        #优化部分
        best_x, best_y = run_opt(args)
        #把history pop一下，去掉最后一个单独的best。 
        #last_T_prob = args.T_prob_history.pop()[0]
        #last_Cu_prob = args.Cu_prob_history.pop()[0]
        #last_Au_prob = args.Au_prob_history.pop()[0]
        #last_Ag_prob = args.Ag_prob_history.pop()[0]
        #last_Cu_balance = args.Cu_balance_history.pop()[0]
        #last_Au_balance = args.Au_balance_history.pop()[0]
        #last_Ag_balance = args.Ag_balance_history.pop()[0]
    
        #从各自的既定位置取出
        best_x = best_x.reshape(-1, args.NUM_OF_TYPES_FOR_GA)
        ga_T = best_x[:, :len(args.obs_T)]
        ga_Cu = best_x[:, len(args.obs_T):len(args.obs_T)+len(args.obs_Cu)]
        ga_Au = best_x[:, len(args.obs_T)+len(args.obs_Cu):len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au)]
        ga_Ag = best_x[:, len(args.obs_T)+len(args.obs_Cu)+len(args.obs_Au):]
        #放回要给web的列表, 这是盘点值，理应都是正的或0
        args.data_all['currentBalanceDry'] = np.round(ga_T.flatten(), 5)
        args.data_all['currentBalancePercentageCu'] = np.round(ga_Cu.flatten(), 5)
        args.data_all['currentBalanceUnitageAu'] = np.round(ga_Au.flatten(), 5)
        args.data_all['currentBalanceUnitageAg'] = np.round(ga_Ag.flatten(), 5)
        #计算消耗、全场回收率
        args, recovery_Ag, recovery_Au, recovery_Cu, recall_Ag, recall_Au, recall_Cu, Cu_balance, Au_balance, Ag_balance = quick_compute(args)
    
        #为了后期方便，加上数据设置页面的值
        for col in args.settings.columns:
            if col not in args.data_all.columns:
                args.data_all[col] = args.settings[col]
        #返回部分
        res_data = miscs.pd_to_res(args.data_all)
        res_data = {
               'list':res_data,
               'parameter':{'recoveryAg':recovery_Ag, \
                            'recoveryAu':recovery_Au, \
                            'recoveryCu':recovery_Cu, \
                            'recallAg':recall_Ag, \
                            'recallAu':recall_Au, \
                            'recallCu':recall_Cu, \
                            'Ag_balance':Ag_balance, \
                            'Au_balance':Au_balance, \
                            'Cu_balance':Cu_balance},
               'set': [0]
                }
        return jsonify(res_data)
    except Exception as e:
        print('Error', e)
        return jsonify({'error': str(e)}), 405



if __name__ == '__main__':
    doc = '金属平衡需要解决‘什么样的真实值最优可能获得目前的观测值’的最大似然问题～求解过程见doc文档，此处从目标函数开始编程。'
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", '--EPOCH', type=int, default=300)
    parser.add_argument("-P", '--POP', type=int, default=3000)
    parser.add_argument('--WEIGHT_T_VOLUME', type=int, default=1)   #volume (T)
    parser.add_argument("--WEIGHT_CU_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_AU_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_AG_PERCENTAGE", type=int, default=1) 
    parser.add_argument("--WEIGHT_BALANCE", type=int, default=1000) # for Cu, Au, Ag
    parser.add_argument("-V", '--IS_VECTOR', action='store_false', default=True)
    parser.add_argument('--PLOT', action='store_true', default=False)
    args = parser.parse_args()

    epsilon = 1e-13
    args.T_prob_history = []
    args.Cu_prob_history = []
    args.Au_prob_history = []
    args.Ag_prob_history = []
    args.Cu_balance_history = []
    args.Au_balance_history = []
    args.Ag_balance_history = []

    app.run(host='0.0.0.0', port=7002, debug=True)

