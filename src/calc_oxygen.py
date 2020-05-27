import argparse
import pandas as pd
import numpy as np
import os,sys,time
import glob
from IPython import embed
import itertools
from time_counter import calc_time
import numpy as np
import copy
#from chempy import Substance
from sympy import *

def calc_oxygen(Mine_in, Quality, Flow=150, Fe2O3_vs_FeO=0.4):
    #Cu in Mine  (T)
    Mine_Cu_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.Cu/100
    Mine_Fe_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.Fe/100
    Mine_S_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.S/100
    Mine_SiO2_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.SiO2/100

    #Matte 冰铜 (%)
    #Quality = Cu% = x*Mass(Cu)*2/(x*Mass(Cu2S) + (1-x)*Mass(FeS))
    x = 11*Quality/(16-9*Quality)   #x为CuS含量
    Matte_Cu = Quality
    Matte_Fe = 7*(1-x)/(11+9*x)
    Matte_S = 4/(11+9*x)

    #Furnace Slag 熔炉渣  (%)
    Slag_Cu = 0.0199
    Slag_S = 0.0045
    Slag_Fe = 0.48
    Slag_SiO2 = 0.24

    #计算冰铜量（Mine_in中所有的Cu转化为Matte的Cu + Slag的Cu, Fe同理）(T)
    # Matte_T*Matte_Cu + Slag_T*Slag_Cu = Mine_Cu_T
    # Matte_T*Matte_Fe + Slag_T*Slag_Fe = Mine_Fe_T
    #定义变量
    x = Symbol('x')
    y = Symbol('y')
    res = solve([x*Matte_Cu + y*Slag_Cu - Mine_Cu_T, x*Matte_Fe + y*Slag_Fe - Mine_Fe_T])
    #所以，修正后的冰铜量以及渣量：
    Matte_T = res[x]
    Slag_T = res[y]
    #冰铜中元素重量、渣中元素重量
    Matte_Fe_T = Matte_T*Matte_Fe
    Matte_S_T = Matte_T*Matte_S
    Slag_Fe_T = Slag_T*Slag_Fe
    Slag_S_T = Slag_T*Slag_S
    
    #(不考虑氧化铜)参加氧化反映的量 = 矿含总量 - 冰铜含总量（剩下的）
    Oxidated_Fe = Mine_Fe_T - Matte_Fe_T
    Oxidated_S = Mine_S_T - Matte_S_T - Slag_S_T
    #2Fe + O2 = 2FeO ;   3Fe + 2O2 = Fe3O4  前者6 后者4
    Oxygen_needed_T_by_Fe = Oxidated_Fe*(1-Fe2O3_vs_FeO)/112*32  + Oxidated_Fe*(Fe2O3_vs_FeO)/168*64
    Oxygen_needed_T_by_S = Oxidated_S*32*1/32
    Oxygen_needed_T = Oxygen_needed_T_by_Fe + Oxygen_needed_T_by_S

    #氧料比：
    Oxygen_Volume = Oxygen_needed_T*1000/1.331058
    OxygenMaterialRatio = Oxygen_Volume/Flow

    return OxygenMaterialRatio.values[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-Quality", '--Quality', type=float, default=0.74)
    parser.add_argument("-Flow", '--Flow', type=float, default=150)
    parser.add_argument("-Fe2O3_vs_FeO", '--Fe2O3_vs_FeO', type=float, default=4/10)
    args = parser.parse_args()

    #Mine in 混合矿输入  
    Mine_in = pd.read_csv("../data/2_ELEMENT_TARGETS.csv")

    OxygenMaterialRatio = calc_oxygen(Mine_in, args.Quality, args.Flow, args.Fe2O3_vs_FeO)
    print("OxygenMaterialRatio", OxygenMaterialRatio)
    

