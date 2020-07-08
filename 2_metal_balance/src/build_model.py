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
import math

def scale_and_precision(x):
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return (magnitude + scale, scale)

def compelete_basic_args(args):
    #measurements:
    args.obs_volume = np.array([1000,2000,3000])
    args.obs_Cu = np.array([0.10, 0.20, 0.30])
    args.obs_Au = np.array([0.001, 0.002, 0.003])
    args.obs_Ag = np.array([0.01, 0.02, 0.03])
    #variances:
    args.obs_variance_wrt_volume = 0.01*args.obs_volume
    args.obs_variance_wrt_Cu = 0.01*args.obs_Cu
    args.obs_variance_wrt_Au = 0.01*args.obs_Au
    args.obs_variance_wrt_Ag = 0.01*args.obs_Ag
    #bounds:
    args.obs_volume_bounds = np.array([args.obs_volume*0.9, args.obs_volume*1.1])
    args.obs_Cu_bounds = np.array([args.obs_Cu*0.9, args.obs_Cu*1.1])
    args.obs_Au_bounds = np.array([args.obs_Au*0.9, args.obs_Au*1.1])
    args.obs_Ag_bounds = np.array([args.obs_Ag*0.9, args.obs_Ag*1.1])
    args.lower_bounds = np.hstack((args.obs_volume_bounds[0,:], args.obs_Cu_bounds[0,:], args.obs_Au_bounds[0,:], args.obs_Ag_bounds[0,:]))
    args.upper_bounds = np.hstack((args.obs_volume_bounds[1,:], args.obs_Cu_bounds[1,:], args.obs_Au_bounds[1,:], args.obs_Ag_bounds[1,:]))
    #others:
    args.NUM_OF_TYPES_FOR_GA = len(args.obs_volume) + len(args.obs_Cu) + len(args.obs_Au) + len(args.obs_Ag)
    args.precisions = 1 / (10**(np.array([scale_and_precision(i)[1] for i in args.lower_bounds])+1))
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
    ga_outcomes = ga_outcomes.reshape(-1, args.NUM_OF_TYPES_FOR_GA)
    ga_volume = ga_outcomes[:, :len(args.obs_volume)]
    ga_Cu = ga_outcomes[:, len(args.obs_volume):len(args.obs_volume)+len(args.obs_Cu)]
    ga_Au = ga_outcomes[:, len(args.obs_volume)+len(args.obs_Cu):len(args.obs_volume)+len(args.obs_Cu)+len(args.obs_Au)]
    ga_Ag = ga_outcomes[:, len(args.obs_volume)+len(args.obs_Cu)+len(args.obs_Au):]
    #Evaluations:
    volume_part = ((args.obs_volume - ga_volume)**2 / args.obs_variance_wrt_volume).sum(axis=1)  
    Cu_part = ((args.obs_Cu - ga_Cu)**2 / args.obs_variance_wrt_Cu).sum(axis=1)  
    Au_part = ((args.obs_Au - ga_Au)**2 / args.obs_variance_wrt_Au).sum(axis=1)  
    Ag_part = ((args.obs_Ag - ga_Ag)**2 / args.obs_variance_wrt_Ag).sum(axis=1)  
    #GA适应度需要最大值，但GA自己取了负数，所以次数直接求最小值即可，不用任何转换
    scores = args.WEIGHT_VOLUME*volume_part + args.WEIGHT_CU_PERCENTAGE*Cu_part + args.WEIGHT_AU_PERCENTAGE*Au_part + args.WEIGHT_AG_PERCENTAGE*Ag_part
    if args.IS_VECTOR:
        scores = scores
    else:
        scores = scores[0]
    return scores

#def run_opt_map(struct):   #map需要，多线程调用GA
#    num = struct[0]
#    args = struct[1]
#    for i in range(100):
#        seed = int(str(time.time()).split('.')[-1])
#        time.sleep(seed/1e9)
#        np.random.seed(seed)
#    print("Process:", num, seed)
#    print("Optimization %s, Dimension %s"%(num, args.NUM_OF_TYPES_FOR_GA))
#    constraint_eq, constraint_ueq = get_constraints(args)
#    GAwrapper.is_vector = args.IS_VECTOR
#    #整数规划，要求某个变量的取值可能个数是2^n，2^n=128, 96+32=128, 则上限为132
#    #考虑一步到位,所有物料参与选择,下限为0
#    ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.POP, max_iter=args.EPOCH, lb=args.lower_bounds, ub=args.upper_bounds, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=args.precisions, prob_mut=0.01)
#    best_gax, best_gay = ga.run()
#    return best_gax, best_gay

def run_opt(args):
#    if args.THREADS != 1:
#        print("Multi thread...Vector mode: %s"%args.IS_VECTOR)
#        blobs = []
#        pool = Pool(processes=int(cpu_count()/2))   #这个固定死，效率最高,跟做多少次没关系
#        struct_list = []
#        for i in range(args.THREADS):  #做threads次
#            struct_list.append([i, args])
#        rs = pool.map(run_opt_map, struct_list) #CORE
#        pool.close()
#        pool.join()
#        return
#    else:
        print("Single thread (Always in metal balancing)... Vector mode: %s"%args.IS_VECTOR)
        print(args)
        constraint_eq, constraint_ueq = get_constraints(args)
        GAwrapper.is_vector = args.IS_VECTOR
        #考虑一步到位,所有物料参与选择,下限为0
        ga = GA(func=GAwrapper, n_dim=args.NUM_OF_TYPES_FOR_GA, size_pop=args.POP, max_iter=args.EPOCH, lb=args.lower_bounds, ub=args.upper_bounds, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=args.precisions, prob_mut=0.01)
        best_gax, best_gay = ga.run()
        embed()
        return

if __name__ == '__main__':
    doc = '金属平衡需要解决‘什么样的真实值最优可能获得目前的观测值’的最大似然问题～求解过程见doc文档，此处从目标函数开始编程。'
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", '--ON_SERVER', action='store_true', default=False)
    parser.add_argument('--COAL_T', type=float, default=1.5)
    parser.add_argument("-E", '--EPOCH', type=int, default=100)
    parser.add_argument("-P", '--POP', type=int, default=10000)
    parser.add_argument('--WEIGHT_VOLUME', type=int, default=1/4)   #volume (T)
    parser.add_argument("--WEIGHT_CU_PERCENTAGE", type=int, default=1/4)  #Cu percentage (%)
    parser.add_argument("--WEIGHT_AU_PERCENTAGE", type=int, default=1/4)  #Au percentage (%)
    parser.add_argument("--WEIGHT_AG_PERCENTAGE", type=int, default=1/4)  #Ag percentage (%)
    parser.add_argument("-M", '--MAX_TYPE_TO_SEARCH', type=int, default=10)
    parser.add_argument("-V", '--IS_VECTOR', action='store_true', default=False)
    args = parser.parse_args()
    args = compelete_basic_args(args)

    manager = Manager()
    normed_dict = manager.dict()
    normed_dict['normed_obj_amount'] = manager.list()

    if args.ON_SERVER:
        app.run(host='0.0.0.0', port=7001, debug=True)
    else:
        run_opt(args) 
        #Mannual:
        sys.exit()
