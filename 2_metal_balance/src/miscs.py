import math
import pandas as pd

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

def pd_to_res(storage):
    res_data = [] 
    key_copying = {'ratio':'calculatePercentage', 'leftover':'inventoryBalance', 'volume_of_storage':'inventory'}
    for i in storage.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
            if key_attr == 'required' or key_attr == 'clean' or key_attr == 'cohesion':   #这个key特殊处理一下true false
                this_dict[key_attr] = True if i[1][this_attr] == 1 else False
            else:
                this_dict[key_attr] = i[1][this_attr]
            if key_attr in key_copying.keys():
                key_attr = key_copying[key_attr]
                this_dict[key_attr] = i[1][this_attr]    #copy to another one
        res_data.append(this_dict) 
    return res_data 


