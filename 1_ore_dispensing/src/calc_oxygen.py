import argparse
import pandas as pd
import numpy as np
import os,sys,time
import glob
from IPython import embed
import itertools
import numpy as np
import copy
#from chempy import Substance
from sympy import *
epsilon = 1e-9
def calc_oxygen(args, Mine_in, debug=False):
    Matte_Cu_Percentage  = args.Matte_Cu_Percentage/100
    Matte_Fe_Percentage  = args.Matte_Fe_Percentage/100
    Matte_S_Percentage   = args.Matte_S_Percentage/100
    Slag_Cu_Percentage   = args.Slag_Cu_Percentage/100
    Slag_S_Percentage    = args.Slag_S_Percentage/100
    Slag_Fe_Percentage   = args.Slag_Fe_Percentage/100
    Slag_SiO2_Percentage = args.Slag_SiO2_Percentage/100

    Oxygen_Peer_Coal     = args.OXYGEN_PEER_COAL
    Coal_T               = args.COAL_T
    Fe_vs_SiO2           = args.Fe_vs_SiO2
    Fe3O4_vs_FeO         = args.Fe3O4_vs_FeO
    Flow                 = args.Flow
    Oxygen_Concentration = args.OXYGEN_CONCENTRATION/100
    RecallRate           = args.RecallRate   #矿的直收率默认0.98
    #print(args.Matte_Cu_Percentage,args.Slag_Cu_Percentage,args.Slag_S_Percentage,args.Slag_Fe_Percentage,args.Slag_SiO2_Percentage,args.Flow , args.Fe3O4_vs_FeO)

    #Cu in Mine  (T)
    Mine_Cu_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.Cu/100*RecallRate
    Mine_Fe_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.Fe/100*RecallRate
    Mine_S_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.S/100*RecallRate
    Mine_SiO2_T = (Flow - Flow*Mine_in.H2O/100)*Mine_in.SiO2/100*RecallRate

    #Matte 冰铜 (%)
    #Matte_Cu_Percentage = Cu% = x*Mass(Cu)*2/(x*Mass(Cu2S) + (1-x)*Mass(FeS))
    #x = 11*Matte_Cu_Percentage/(16-9*Matte_Cu_Percentage)   #x为CuS含量, Matte_Cu_Percentage 0.74
    #calc_Matte_Cu_Percentage = Matte_Cu_Percentage
    #calc_Matte_Fe_Percentage = 7*(1-x)/(11+9*x)
    #calc_Matte_S_Percentage = 4/(11+9*x)
    #print("calc:", calc_Matte_Fe_Percentage, calc_Matte_S_Percentage, calc_Matte_Cu_Percentage)

    #Furnace Slag 熔炉渣  (%)
    #Slag_Cu_Percentage = 0.0199
    #Slag_S_Percentage = 0.0045
    #Slag_Fe_Percentage = 0.48
    #Slag_SiO2_Percentage = 0.24

    #计算冰铜量（Mine_in中所有的Cu转化为Matte的Cu + Slag的Cu, Fe同理）(T)
    # Matte_T*Matte_Cu_Percentage + Slag_T*Slag_Cu_Percentage = Mine_Cu_T
    # Matte_T*Matte_Fe_Percentage + Slag_T*Slag_Fe_Percentage = Mine_Fe_T
    # Matte_T = (Mine_Cu_T - Slag_T*Slag_Cu_Percentage)/Matte_Cu_Percentage
    # (Mine_Cu_T - Slag_T*Slag_Cu_Percentage)*Matte_Fe_Percentage/Matte_Cu_Percentage + Slag_T*Slag_Fe_Percentage = Mine_Fe_T
    # Slag_T = (Mine_Fe_T - Mine_Cu_T*Matte_Fe_Percentage/Matte_Cu_Percentage)/(Slag_Fe_Percentage- Slag_Cu_Percentage*Matte_Fe_Percentage/Matte_Cu_Percentage)
    #
    #定义变量
    #x = Symbol('x')
    #y = Symbol('y')
    #res = solve([x*Matte_Cu_Percentage + y*Slag_Cu_Percentage - Mine_Cu_T, x*Matte_Fe_Percentage + y*Slag_Fe_Percentage - Mine_Fe_T])
    ##所以，修正后的冰铜量以及渣量：
    #Matte_T = res[x]
    #Slag_T = res[y]
    #手动解方程（更快）
    Slag_T = (Mine_Fe_T - Mine_Cu_T*Matte_Fe_Percentage/Matte_Cu_Percentage)/(Slag_Fe_Percentage- Slag_Cu_Percentage*Matte_Fe_Percentage/Matte_Cu_Percentage)
    Matte_T = (Mine_Cu_T - Slag_T*Slag_Cu_Percentage)/Matte_Cu_Percentage
    #冰铜中元素重量、渣中元素重量
    Matte_Fe_T = Matte_T*Matte_Fe_Percentage
    Matte_S_T = Matte_T*Matte_S_Percentage
    Slag_Fe_T = Slag_T*Slag_Fe_Percentage
    Slag_S_T = Slag_T*Slag_S_Percentage
    
    #(不考虑氧化铜)参加氧化反映的量 = 矿含总量 - 冰铜含总量（剩下的）
    Oxidated_Fe_T = Mine_Fe_T - Matte_Fe_T
    Oxidated_S_T = Mine_S_T - Matte_S_T - Slag_S_T
    #2Fe + O2 = 2FeO ;   3Fe + 2O2 = Fe3O4  前者0.4 后者0.6
    Oxygen_needed_T_by_Fe = Oxidated_Fe_T*(1-1/(1+1/(Fe3O4_vs_FeO+epsilon)))/112*32  + Oxidated_Fe_T*(1/(1+1/(Fe3O4_vs_FeO+epsilon)))/168*64
    Oxygen_needed_T_by_S = Oxidated_S_T*32*1/32
    Oxygen_needed_T = Oxygen_needed_T_by_Fe + Oxygen_needed_T_by_S

    #氧料比：
    Oxygen_Volume = Oxygen_needed_T*1000/32*22.4
    OxygenMaterialRatio = Oxygen_Volume/Flow
    if debug:
        embed()

    #一次风量：
    Wind_Flux = (Oxygen_Peer_Coal * Coal_T + Oxygen_Volume) / Oxygen_Concentration

    #石英石量(ratio):
    SiO2_T = ((Oxidated_Fe_T/Fe_vs_SiO2) - Mine_SiO2_T)/Flow*100

    #返回：氧料比，冰铜量，渣量，一次风量，石英石量
    return OxygenMaterialRatio.values[0], Matte_T.values[0], Slag_T.values[0], Wind_Flux.values[0], SiO2_T.values[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-Matte_Cu_Percentage", '--Matte_Cu_Percentage', type=float, default=0.74)
    parser.add_argument("-Flow", '--Flow', type=float, default=150)
    parser.add_argument("-Fe3O4_vs_FeO", '--Fe3O4_vs_FeO', type=float, default=4/6)
    args = parser.parse_args()

    #Mine in 混合矿输入  
    Mine_in = pd.read_csv("../data/2_ELEMENT_TARGETS.csv")

    OxygenMaterialRatio = calc_oxygen(Mine_in, args.Matte_Cu_Percentage)
    print("OxygenMaterialRatio", OxygenMaterialRatio)
    

