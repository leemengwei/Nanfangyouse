import pandas as pd
import numpy as np
import os,sys,time
import glob
from IPython import embed
import matplotlib.pyplot as plt
import datetime
from scipy.special import comb, perm


ELEMENTS = ['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O']
which_is_time = 0
which_is_percentage = 2 

def C(n,k):  
    #import operator
    #return reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))  
    out = comb(n, k)
    return out

def get_storage():
    ingredient_storage = pd.read_csv("../data/0_ingredient_storage.csv", index_col='name')
    #str to datetime
    for row_idx,row in enumerate(ingredient_storage.iterrows()):
        ingredient_storage.iloc[row_idx, which_is_time] = datetime.datetime.strptime(row[1].when_comes_in, "%Y/%m/%d %H:%M")
    return ingredient_storage

def get_element_targets():
    element_targets = pd.read_csv("../data/1_element_targets.csv")
    return element_targets

def get_solution():
    solution = pd.read_csv("../data/2_solution.csv", index_col='name')
    #str to percentage
    for row_idx,row in enumerate(solution.iterrows()):
        solution.iloc[row_idx, which_is_percentage] = int(row[1].ratio.strip("%"))/100
    return solution

def mixing(ingredient_storage, solution):
    element_output = pd.DataFrame(np.array([0]*len(ELEMENTS)).reshape(1,-1), columns = ELEMENTS)
    for this_type in solution.index:                                               
        element_output += solution.loc[this_type].ratio * ingredient_storage.loc[this_type][ELEMENTS]
    #which one insufficient first?  #TODO: to check: AMOUNT columns from solution equal to AMOUNT columns from storage?
    first_insufficient_type = solution[solution.volume_of_storage*solution.ratio==min(solution.volume_of_storage*solution.ratio)].index[0]
    consumed_amount = (solution.ratio/solution.loc[first_insufficient_type].ratio)*solution.loc[first_insufficient_type].volume_of_storage
    #after consumed, leftovers are:
    solution['consumed_amount'] = consumed_amount
    solution['leftover'] = solution.volume_of_storage - (solution.ratio/solution.loc[first_insufficient_type].ratio)*solution.loc[first_insufficient_type].volume_of_storage
    #print(solution, '\n', element_output)
    return solution, element_output

if __name__ == '__main__':
    print("Start....")

    #获取库存
    ingredient_storage = get_storage()

    #获取目前的solution
    solution = get_solution()

    #根据目前的solution混合，得到混合结果
    solution, element_output = mixing(ingredient_storage, solution)

    #获取元素配比目标
    element_targets = get_element_targets()

    #根据混合结果得到Objectives:
    obj_amount = solution.consumed_amount         #越大越好
    obj_leftover = solution.leftover              #越小越好
    obj_elements = abs(element_targets - element_output)  #越小越好

    #Objectives无量纲化：
    normed_obj_amount = obj_amount/ingredient_storage.volume_of_storage    
    normed_obj_leftover = obj_leftover/solution.volume_of_storage 
    normed_obj_elements = obj_elements/element_targets    

    #Multi-Obj to Single_obj:
    alpha = 1/3
    beta = 1/3
    gama = 1-alpha-beta
    objective_function = alpha*normed_obj_amount.sum() + (-1)*beta*normed_obj_leftover.sum() + (-1)*gama*normed_obj_elements.sum(axis=1)
    objective_function = objective_function.values

    print("SOLUTION yields:", objective_function)
    embed()




